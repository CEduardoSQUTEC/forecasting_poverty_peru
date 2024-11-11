# Contributing to Poverty Forecasting in Peru

Thank you for considering contributing to Poverty Forecasting in Peru! To ensure a smooth contribution process, please follow the guidelines outlined in this document.

## How to Contribute

1. **Fork the Repository**: Create a personal copy of the repository.

2. **Create a Branch**: Use a descriptive branch name for your changes. For example:

    - `feature/add-new-feature`
    - `fix/resolve-bug-issue`

3. **Make Your Changes**: Implement your changes or fixes.

4. **Commit Your Changes**: Follow the [Conventional Commits](https://www.conventionalcommits.org) specification when writing your commit messages.

## Commit Message Guidelines

We use [**Conventional Commits**](https://www.conventionalcommits.org) to structure our commit messages. This helps in understanding the history of changes and automating versioning and changelog generation.

### Commit Message Format

A commit message consists of a **header**, an optional **body**, and an optional **footer**. The header has a special format that includes a type, an optional scope, and a subject.

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Types

Here are the types you can use:

- `feat`: A new feature for the user.

- `fix`: A bug fix for the user.

- `docs`: Documentation only changes.

- `style`: Changes that do not affect the meaning of the code (white-space, formatting, etc.).

- `refactor`: A code change that neither fixes a bug nor adds a feature.

- `perf`: A code change that improves performance.

- `test`: Adding missing tests or correcting existing tests.

- `chore`: Changes to the build process or auxiliary tools and libraries such as documentation generation.

#### Body

The body of the commit message should provide additional context about the change. It can include:

- Why the change was made.

- Any relevant information regarding implementation details.

#### Footer

The footer can be used to reference issues or breaking changes:

```
BREAKING CHANGE: The login API has changed from /login to /auth/login.
```