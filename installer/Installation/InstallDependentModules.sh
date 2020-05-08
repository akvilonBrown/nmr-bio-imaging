#!/bin/bash
#path to the Python dependents modules.
FILE_PATH='./dependent_modules.txt'

installModules() {

  if [ ! -f $1 ]; then
    echo "File 'dependent_modules.txt' was not found! Please add this file with dependent modules!"
    return
  fi

  while IFS= read -r line
    do
      if pip3 freeze | grep $line=;
      then
        echo "++++ $line module has been already installed. ++++"
      else
        if [ $line == 'tensorflow' ];
        then
           #install tensorflow version 1
           pip3 install "tensorflow>=1.15,<2.0"
        else
           pip3 install $line
        fi
      fi
  done < "$1"

}

installModules "$FILE_PATH"
