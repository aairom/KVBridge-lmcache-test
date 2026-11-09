# Git  
   - Always provide a ".gitignore" file which filters/ignores any ".env" files or any folders whichs' names start with "_" (underscore) to be pushed to GitHub (e.g.: _sources/, _images/, _docs/... ).
   - Add the following files to .gitignore if they exist in the project: *.onnx, *.onnx.data, *.so, *.tgz, *.tar.gz, *.zip, *.dylib
   - "node_modules" folder should be in .gitignore and not be sent to Github
   - In the ".gitignore" add an entry to avoid sending ".playwright-mcp" folder to remote repository
   - "input" folder should be synced with Github, not it's content which should be set in .gitignore
   -  "output" folder should be synced with Github, not it's content which should be set in .gitignore
   -  Exclude AGENTS.md to be pushed to github in .gitignore
   -  Exclude ".bob/mcp.json" to be pushed to github in .gitignore