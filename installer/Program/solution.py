import solutionModule
import sys

SOURCE_FOLDER = sys.argv[1]
RESULT_FOLDER = sys.argv[2]
print('Source folder >>>', SOURCE_FOLDER)
print('Destination folder >>>', RESULT_FOLDER)
solutionModule.runModel(SOURCE_FOLDER, RESULT_FOLDER)
