"""Model-independent helpers; imports other libs modules only, never models/, pipelines/ or nodes/.
At module level only the standard library, torch and numpy; anything else (PIL, scipy, cv2, av,
folder_paths) inside the function that uses it."""
