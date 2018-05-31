import os
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(description='Runs detector against testdata')
    parser.add_argument('--test-data',
        help="Path to test data directory.  Default: './originalDataSet'")

    args = parser.parse_args()

    basedir = './originalDataSet/'
    if args.test_data:
        basedir = args.test_data

    userNumber = 0
    for folderName in sorted(os.listdir(basedir)):
        fullPath = basedir + '/' + (folderName)
        if '_' in folderName:
            continue
        if os.path.isdir(fullPath) :
            userName = 'User_' + str(userNumber)
            newFolderName = basedir+ userName
            os.rename(fullPath,newFolderName)
            fileNumber = 0
            for curFile in sorted(os.listdir(newFolderName)):
                oldPath = newFolderName + '/' + curFile
                if not os.path.isdir(oldPath):
                    ext = curFile.split('.')[len(curFile.split('.'))-1]
                    newFileName = newFolderName+ '/' + userName + '.' + str(fileNumber) + '.' + ext
                    os.rename(oldPath,newFileName)
                    fileNumber = fileNumber+1
                else:
                    shutil.rmtree(oldPath)
            userNumber = userNumber + 1


if __name__ == "__main__":
    main()