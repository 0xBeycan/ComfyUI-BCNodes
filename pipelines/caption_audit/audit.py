"""Caption Audit plumbing: the caption_audit package guard, AuditArgs, the allowed roots and
resolve_dir, build_args, run_audit, the text/JSON reports and the caption-file fingerprint.
"""

import os


def _ca():
    try:
        import caption_audit
        import caption_audit.cli  # noqa: F401  (submodule, not re-exported)
    except ImportError as exc:
        raise ImportError("Caption Audit needs the `caption-audit` package: pip install 'caption-audit>=1.3'") from exc
    return caption_audit


def package_version():
    return getattr(_ca(), "__version__", "")


# --- audit plumbing ---------------------------------------------------------

class AuditArgs:
    """An argparse.Namespace stand-in.

    `analyze()` and the renderers only ever *read* attributes, so a plain
    object with the same field names is a complete substitute — no argv
    round-trip, no subprocess, no temp files.
    """

    __slots__ = ("dir", "images_dir", "recursive", "trigger", "class_word",
                 "fuse", "critical_threshold", "warn_threshold",
                 "info_threshold", "ngram_max", "jaccard", "no_stopwords",
                 "list_files", "max_rows", "max_interpret", "no_color",
                 "format", "out", "compare", "positional_dir")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return "AuditArgs(%s)" % ", ".join(
            "%s=%r" % (k, getattr(self, k)) for k in self.__slots__)


def split_terms(text):
    """'red scarf, studio lighting' -> ['red scarf', 'studio lighting'].

    Comma-separated because a term is itself allowed to contain spaces.
    """
    if not text:
        return []
    return [t.strip() for t in str(text).split(",") if t.strip()]


# Widget values come out of the workflow JSON, so a path the node resolves and
# nothing else would let any graph you open read .txt sidecars anywhere on the
# machine and hand their names and contents straight back out through
# report_text / report_json. The caption set is therefore confined to the
# ComfyUI tree plus whatever roots the host - never the graph - names in
# BC_CAPTION_ROOTS: datasets live outside ComfyUI, and that is the one place
# the person running the server can say so.
ROOTS_ENV = "BC_CAPTION_ROOTS"


def allowed_roots():
    """Directories a caption widget may point at, resolved and de-duplicated."""
    roots = []
    try:
        import folder_paths
        roots.append(getattr(folder_paths, "base_path", None)
                     or os.path.dirname(folder_paths.models_dir))
        for getter in ("get_input_directory", "get_output_directory", "get_user_directory"):
            fn = getattr(folder_paths, getter, None)
            if fn is not None:
                roots.append(fn())
    except Exception:  # outside ComfyUI (tests, a bare import): the cwd only
        roots.append(os.getcwd())
    roots.extend(os.environ.get(ROOTS_ENV, "").split(os.pathsep))

    out = []
    for root in roots:
        root = str(root or "").strip()
        if not root:
            continue
        real = os.path.realpath(os.path.expanduser(root))
        if real not in out:
            out.append(real)
    return out


def within(path, root):
    """Is `path` inside `root`? Both must already be realpath'd."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


def resolve_dir(path):
    """Expand ~ and relative paths; '' means the ComfyUI process cwd.

    realpath, not abspath: '..' segments and symlinks are collapsed before the
    result is checked against allowed_roots(). A path outside them raises
    ValueError naming the roots, which the node draws as an error card like any
    other audit failure.
    """
    raw = str(path or "").strip()
    resolved = os.path.realpath(os.path.expanduser(raw) or os.getcwd())
    roots = allowed_roots()
    if any(within(resolved, root) for root in roots):
        return resolved
    raise ValueError(
        "%s is outside the folders this node may read (%s). Put the dataset "
        "under one of them, or name its root in the %s environment variable "
        "('%s'-separated) and restart ComfyUI."
        % (resolved, ", ".join(roots), ROOTS_ENV, os.pathsep))


def build_args(directory, trigger="", images_dir="", class_words="", fuse="",
               critical_threshold=0.85, warn_threshold=0.60,
               info_threshold=0.35, ngram_max=3, jaccard=0.9,
               no_stopwords=False, recursive=False, max_rows=200):
    """Every field `analyze()` and the renderers read, and nothing else."""
    imgs = str(images_dir or "").strip()
    return AuditArgs(
        dir=directory,
        positional_dir=None,
        images_dir=resolve_dir(imgs) if imgs else None,
        recursive=bool(recursive),
        trigger=str(trigger or "").strip() or None,
        class_word=split_terms(class_words),
        fuse=split_terms(fuse),
        critical_threshold=float(critical_threshold),
        warn_threshold=float(warn_threshold),
        info_threshold=float(info_threshold),
        ngram_max=int(ngram_max),
        jaccard=float(jaccard),
        no_stopwords=bool(no_stopwords),
        # report-only knobs: generous, since the text/JSON outputs are the
        # complete record and the card does its own fixed-height trimming
        list_files=10,
        max_rows=int(max_rows),
        max_interpret=12,
        no_color=True,
        format="term",
        out=None,
        compare=None,
    )


def run_audit(directory, args):
    """analyze() with the thresholds validated the way the CLI validates them.

    Returns the report dict, or raises ValueError with a message meant to be
    read on the card.
    """
    if not (0 < args.warn_threshold <= args.critical_threshold <= 1.0):
        raise ValueError("need 0 < warn_threshold <= critical_threshold <= 1.0")
    if not (0 < args.info_threshold <= args.warn_threshold):
        raise ValueError("need 0 < info_threshold <= warn_threshold")
    if not os.path.isdir(directory):
        raise ValueError("not a directory: %s" % directory)
    if args.images_dir and not os.path.isdir(args.images_dir):
        raise ValueError("not a directory: %s" % args.images_dir)

    rep = _ca().cli.analyze(directory, args)
    if rep is None:
        raise ValueError("no .txt caption files in %s (or no trigger could be "
                         "inferred - set 'trigger')" % directory)
    return rep


def report_text(rep, args):
    """The plain-text terminal report, ANSI disabled."""
    from caption_audit.report import Palette, render_term_report
    return render_term_report(rep, Palette(False), args)


def report_json(rep):
    from caption_audit.report import to_json
    return to_json(rep)


def dir_fingerprint(directory, recursive=False):
    """(path, mtime, size) digest of the caption files under `directory`.

    Used by IS_CHANGED so editing captions re-runs the audit, while an
    untouched dataset still hits ComfyUI's execution cache.
    """
    import hashlib

    h = hashlib.sha256()
    directory = str(directory or "")
    if not os.path.isdir(directory):
        return directory
    try:
        if recursive:
            walker = ((dp, fn) for dp, _dn, fns in os.walk(directory) for fn in fns)
        else:
            walker = ((directory, fn) for fn in os.listdir(directory))
        for dirpath, fn in sorted(walker):
            if not fn.lower().endswith(".txt") or fn.startswith("."):
                continue
            p = os.path.join(dirpath, fn)
            st = os.stat(p)
            h.update(("%s|%d|%d;" % (p, st.st_mtime_ns, st.st_size)).encode("utf-8"))
    except OSError as exc:
        return "%s|error:%s" % (directory, exc)
    return h.hexdigest()
