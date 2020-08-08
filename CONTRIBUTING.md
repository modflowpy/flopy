Contributing
============

Contributions to FloPy are welcome from the community. As a contributor, here are the guidelines we would like you to follow:

 - [Code of Conduct](#coc)
 - [Issues and Bugs](#issue)
 - [Feature Requests](#feature)
 - [Submission Guidelines](#submit)
 - [Coding Rules](#rules)
 - [Commit Message Guidelines](#commit)

## <a name="coc"></a> Code of Conduct
Help us keep FloPy open and inclusive. Please read and follow our [Code of Conduct][coc].

## <a name="issue"></a> Found a Bug?
If you find a bug in the source code, you can help us by
[submitting an issue](#submit-issue) to our [GitHub Repository][github]. Even better, you can [submit a Pull Request](#submit-pr) with a fix.

## <a name="feature"></a> Missing a Feature?
You can *request* a new feature by [submitting an issue](#submit-issue) to our GitHub Repository. If you would like to *implement* a new feature, please submit an issue with a proposal for your work first, to be sure that we can use it. Please consider what kind of change it is:

* For a **Major Feature**, first open an issue and outline your proposal so that it can be
discussed. This will also allow us to better coordinate our efforts, prevent duplication of work,
and help you to craft the change so that it is successfully accepted into the project.
* **Small Features** can be crafted and directly [submitted as a Pull Request](#submit-pr).

## <a name="submit"></a> Submission Guidelines

### <a name="submit-issue"></a> Submitting an Issue

Before you submit an issue, please search the issue tracker, maybe an issue for your problem already exists and the discussion might inform you of workarounds readily available.

We want to fix all the issues as soon as possible, but before fixing a bug we need to reproduce and confirm it. In order to reproduce bugs, we will systematically ask you to provide a minimal, complete, and verifiable example. Having a minimal, complete, and verifiable example gives us a wealth of important information without going back & forth to you with additional questions like:

- version of FloPy used
- and most importantly - a use-case that fails (ideally an example that uses flopy to generate FloPy input files - see `t0**_test.py` python scripts in the `autotest/` directory)

We will be insisting on a minimal, complete, and verifiable example in order to save maintainers time and ultimately be able to fix more bugs. We understand that sometimes it might be hard to extract essentials bits of code from a larger code-base but we really need to isolate the problem before we can fix it.

Unfortunately, we are not able to investigate / fix bugs without a minimal, complete, and verifiable example, so if we don't hear back from you we are going to close an issue that doesn't have enough info to be reproduced.

You can file new issues by filling out our [new issue form](https://github.com/modflowpy/flopy/issues/new).


### <a name="submit-pr"></a> Submitting a Pull Request (PR)
Before you submit your Pull Request (PR) consider the following guidelines:

1. Search [GitHub](https://github.com/modflowpy/flopy/pulls) for an open or closed PR that relates to your submission. You don't want to duplicate effort.
1. Fork the modflowpy/flopy repo.
1. Make your changes in a new git branch:

     ```shell
     git checkout -b my-fix-branch develop
     ```

1. Create your patch, **including appropriate test cases**.
1. Run the [black formatter](https://github.com/psf/black) on Flopy source files from the git repository root directory using:

   ```shell
   black -l 79 ./flopy
   ```
   Note: Pull Requests must pass black format checks run on the [GitHub actions](https://github.com/modflowpy/flopy/actions) (*linting*) before they will be accepted. The black formatter can be installed using [`pip`](https://pypi.org/project/black/) and [`conda`](https://anaconda.org/conda-forge/black). 
   
1. Run the full FloPy test suite and ensure that all tests pass:

    ```shell
    cd autotest
    nosetests -v build_exes.py
    nosetests -v
    ```
   Note: the FloPy test suite requires the [nosetests](https://pypi.org/project/nose/) and [pymake](https://github.com/modflowpy/pymake) python packages. All the FloPy dependencies must also be installed for the tests to pass.

1. Commit your changes using a descriptive commit message that follows our
  [commit message conventions](#commit). Adherence to these conventions
  is necessary because release notes are automatically generated from these messages.

     ```shell
     git commit -a
     ```
   Note: the optional commit `-a` command line option will automatically "add" and "rm" edited files.

1. Push your branch to GitHub:

    ```shell
    git push origin my-fix-branch
    ```

1. In GitHub, send a pull request to `flopy:develop`.
* If we suggest changes then:
  * Make the required updates.
  * Re-run the FloPy test suites, in the autotest directory, to ensure tests are still passing.
  * Rebase your branch and force push to your GitHub repository (this will update your Pull Request):

    ```shell
    git rebase develop -i
    git push -f
    ```

That's it! Thank you for your contribution!

#### After your pull request is merged

After your pull request is merged, you can safely delete your branch and pull the changes
from the main (upstream) repository:

* Delete the remote branch on GitHub either through the GitHub web UI or your local shell as follows:

    ```shell
    git push origin --delete my-fix-branch
    ```

* Check out the develop branch:

    ```shell
    git checkout develop -f
    ```

* Delete the local branch:

    ```shell
    git branch -D my-fix-branch
    ```

* Update your develop with the latest upstream version:

    ```shell
    git pull --ff upstream develop
    ```

## <a name="rules"></a> Coding Rules
To ensure consistency throughout the source code, keep these rules in mind as you are working:

* All features or bug fixes **must be tested** by one or more specs (unit-tests and/or integration/regression tests).

## <a name="commit"></a> Commit Message Guidelines

We have very precise rules over how our git commit messages can be formatted.  This leads to **more
readable messages** that are easy to follow when looking through the **project history**.  But also,
we use the git commit messages to **generate the FloPy change log**.

### Commit Message Format
Each commit message consists of a **header**, a **body** and a **footer**.  The header has a special
format that includes a **type**, a **scope** and a **subject**:

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

The **header** is mandatory and the **scope** of the header is optional.

Any line of the commit message cannot be longer 100 characters! This allows the message to be easier
to read on GitHub as well as in various git tools.

The footer should contain a [closing reference to an issue](https://help.github.com/articles/closing-issues-using-keywords/) if any.

Samples: (even more [samples](https://github.com/modflowpy/flopy/commits/develop))

```
docs(changelog): update changelog to beta.5
```
```
fix(release): need to depend alslkj askjalj lhjfjepo kjpodep

The version in our lakjoifejw jiej kdjijeqw kjdwjopj lkl kopfcqiw cakd kjkfje mmsm.
```

### Revert
If the commit reverts a previous commit, it should begin with `revert: `, followed by the header of the reverted commit. In the body it should say: `This reverts commit <hash>.`, where the hash is the SHA of the commit being reverted.

### Type
Must be one of the following:

* **ci**: Changes to our CI configuration files and scripts (example scopes: Travis)
* **docs**: Documentation only changes
* **feat**: A new feature
* **fix**: A bug fix
* **perf**: A code change that improves performance
* **refactor**: A code change that neither fixes a bug nor adds a feature
* **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
* **test**: Adding missing tests or correcting existing tests

### Scope
The scope should be the name of the FloPy module/class affected (as perceived by the person reading the changelog generated from commit messages.

There are currently a few exceptions to the "use module/class name" rule:

* **release**: used when updating files prior to a release
* **releasenotes**: used for updating the release notes
* **readme**: used for updating the release notes in README.md
* **changelog**: used for updating the release notes in CHANGELOG.md
* none/empty string: useful for `style`, `test` and `refactor` changes that are done across all
  packages (e.g. `style: add missing semicolons`) and for docs changes that are not related to a
  specific package (e.g. `docs: fix typo in tutorial`).

### Subject
The subject contains a succinct description of the change:

* use the imperative, present tense: "change" not "changed" nor "changes"
* don't capitalize the first letter
* no dot (.) at the end

### Body
Just as in the **subject**, use the imperative, present tense: "change" not "changed" nor "changes".
The body should include the motivation for the change and contrast this with previous behavior.

### Footer
The footer should contain any information about **Breaking Changes** and is also the place to reference GitHub issues that this commit **Closes**.

**Breaking Changes** should start with the word `BREAKING CHANGE:` with a space or two newlines. The rest of the commit message is then used for this.


[coc]: https://github.com/modflowpy/flopy/blob/develop/CODE_OF_CONDUCT.md
[github]: https://github.com/modflowpy/flopy
[stackoverflow]: http://stackoverflow.com/questions/tagged/flopy
