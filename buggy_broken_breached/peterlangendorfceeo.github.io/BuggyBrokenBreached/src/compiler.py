import os

def compile_project_files(output_filename="allcode.txt"):
    # Target extensions to compile
    target_extensions = ('.js', '.py', '.css', '.txt')
    
    # Folders to completely ignore (keeps things fast and clean)
    ignore_folders = {'.git', '__pycache__', 'node_modules', '.vscode'}
    
    # Get the absolute path of the output file so we don't accidentally read it
    output_path = os.path.abspath(output_filename)
    
    count = 0

    print(f"Scanning for {target_extensions} files...")
    
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        # os.walk traverses the current directory and all subdirectories
        for root, dirs, files in os.walk('.'):
            
            # Modify the 'dirs' list in-place to skip ignored folders
            dirs[:] = [d for d in dirs if d not in ignore_folders]
            
            for file in files:
                file_path = os.path.abspath(os.path.join(root, file))
                
                # Never read the output file itself, or this script (if it's named compiler.py)
                if file_path == output_path or file == "compiler.py":
                    continue
                
                if file.endswith(target_extensions):
                    # Get a clean relative path for the header (e.g., "js/state.js")
                    rel_path = os.path.relpath(file_path, '.')
                    
                    # Write the separation header
                    outfile.write(f"\n{'='*60}\n")
                    outfile.write(f"FILE: {rel_path}\n")
                    outfile.write(f"{'='*60}\n\n")
                    
                    # Read the file and write its contents
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                        count += 1
                        print(f"Added: {rel_path}")
                    except Exception as e:
                        outfile.write(f"// ERROR READING FILE: {e}\n")
                        print(f"Failed to read {rel_path}: {e}")
                        
    print(f"\nSuccess! Compiled {count} files into '{output_filename}'.")

if __name__ == "__main__":
    compile_project_files()