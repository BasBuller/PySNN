# Contributing Guidelines

Contributing to PySNN follows the standard protocol:
1. Clone the PySNN repository
2. Make changes on a separate development branch
3. Push the branch to the remote repository
4. Open a new pull request
5. The pull request is reviewed by one of the maintainers

## Setup development branch

Clone the PySNN library to your local machine using the following command:

    git clone https://github.com/BasBuller/PySNN.git

## Make your changes

Your changes should contain logically grouped commits, with each commit containing work on a single subject. Add clear and concise comments that describe the code. In case a new function or class is added also add a docstring that is formatted in __ReStructuredText__ (rst).

## Push the branch to the remote

Before pushing your changes to the remote, make sure your commits are as clean as possible. Also, be sure to update your working branch to match the latest master branch. This can by done by rebasing your changes on the master branch by using the following command (while working on your development branch):

    git pull --rebase origin master

Next, push to the remote repository with the following command:

    git push origin [branch name]

## Open a pull request

Lastly, open a new pull request on the github webpage. Please add a short description of the intention behind the request. If you want a specific reviewer to look at your code, please request so at in the panel on the right when making the pull request.

## Wrapping up

Once the reviewer is done, your pull request will be merged if everything is looking good. Otherwise, the reviewer will request the kind of changes he/she would like to see. Once the updates are in place, the pull request can be updated.
