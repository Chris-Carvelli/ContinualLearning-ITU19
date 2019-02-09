"""
This is a test document to test out the possibility of using git commits when saving data from training runs so
the experiments can be re-run with the same settings/code as they the data was generated
"""
import os

import git

from git import Repo

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
# print(ROOT_DIR)
# print(os.getcwd())

repo = Repo('C:/Users/Frost/Documents/Dropbox/ITU/KurserITU/Thesis Project/Repository/tests/rasf/TestRepo')
print(repo)

assert not repo.is_dirty()  # check the dirty state
print(repo.untracked_files)             # retrieve a list of untracked files


print(repo.head)
print(repo.head.commit)

# secondary : 18c5b5f169e9ce25248acc87c9c4d74a00617ddd
# master 2 : 4583abf1f4967cc0922c45ad9be4acc38f2e3a23
# master 1 : 7ab1c6a3d9b2ae5098837324d22a4c689bb394cc
