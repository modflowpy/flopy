Instructions for using git-hooks to automatically update FloPy `build` in `version.py`
-----------------------------------------------

## Update files

1. Delete `pre-commit.sample` file in `.git/hooks`.
2. Copy `pre-commit.sh` in root directory to `.git/hooks` directory.
3. Remove `.sh` extension from `pre-commit.sh` in `.git/hooks` directory.
4. Make sure `.git/hooks/pre-commit` is executable using `chmod +x .git/hooks/pre-commit`.

## Reset SourceTree to use system git

1.  SourceTree will use git-hooks if the Embedded Git is git 2.9+. SourceTree version 2.5.3 uses git 2.10.0.
2. If your version of SourceTree is using a version earlier than 2.9 then modify the SourceTree preferences to use the system version of git. All you have to do is got to `SourceTree -> Preferences -> Git` and choose `Use System Git` which can be found at `/usr/bin/git/`. See [https://medium.com/@onmyway133/sourcetree-and-pre-commit-hook-52545f22fe10](https://medium.com/@onmyway133/sourcetree-and-pre-commit-hook-52545f22fe10) for additional information. 

