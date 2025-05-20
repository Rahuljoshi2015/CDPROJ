import sys
import os
import subprocess

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 1 and os.path.isdir(args[0]):
        folder = args[0]
        m_files = [f for f in os.listdir(folder) if f.endswith(".m")]
        if not m_files:
            print(f"No .m files found in folder: {folder}")
        for file in m_files:
            filepath = os.path.join(folder, file)
            print(f"Running ast_generator.py on {filepath}")
            # Run ast_generator.py on each file separately
            subprocess.run(["python", "ast_generator.py", filepath])
    else:
        print("Please provide exactly one folder path as argument")
