[![Build/Release Python Package](https://github.com/calico/github-template-python-library/actions/workflows/release-new-version.yml/badge.svg?branch=main)](https://github.com/calico/github-template-python-library/actions/workflows/release-new-version.yml)
[![Python formatting and tests](https://github.com/calico/github-template-python-library/actions/workflows/run-tests-formatting.yml/badge.svg?branch=main)](https://github.com/calico/github-template-python-library/actions/workflows/run-tests-formatting.yml)
[![Validate prettier formatting](https://github.com/calico/github-template-python-library/actions/workflows/check-prettier-formatting.yml/badge.svg?branch=main)](https://github.com/calico/github-template-python-library/actions/workflows/check-prettier-formatting.yml)

# Github Template for Calico's Python Library

[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)

Create internal Python packages using this template.

## Repo setup

1. Create a new `calicolabs-{python-library-name}` Github repo using this template. The new repo name must have a `calicolabs-` prefix.
2. Navigate to **Actions** > `Set up new Python library repo` > Run workflow. The `packageName` input cannot have spaces or hyphens. If you need to use multiple words, separate them by underscores. This name can be anything that makes sense to you. It will be what replaces the package_name folder under the src directory. For example, your input at this step will be used in importing like `from package_name.some_sub_folder.some_file import some_function` when you write your code.
3. This will create a new PR and upon review and approval, merge into `main`. Your Python library repo is now ready to use!

## Additional setup

### Update the following parameters in the `setup.cfg` file.

Uncomment the following lines and add your package dependencies.
Where possible, please add `~=` instead of `==`

```
;install_requires =
;    package~=3.17.0
;    package2~=3.15.1
```

## Push the versioned package to Calico-PyPI (our internal repository)

Note: We follow **[Semantic Versioning](https://semver.org/)** for versioning Python packages.

1. Navigate to **Actions** > `Build/Release Python Package` > Run workflow. This will build and push your versioned package to Calico-PyPI.

## Formatting

### Prettier

- Install and run `prettier` to maintain consistent YAML/Markdown/JSON formatting.
  - In repo root: `npm install prettier@2.7.1`
  - Reformat repo after changes: `npx prettier --write .`

### Black

- Install and run `black` to maintain consistent Python formatting.
  - In repo root: `pip install black`
  - Reformat repo after changes: `black .`

### Ruff

- Install and run `ruff` to maintain consistent Python formatting.
  - In repo root: `pip install ruff`
  - Reformat repo after changes: `ruff check .`
