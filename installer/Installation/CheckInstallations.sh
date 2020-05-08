
#!/bin/bash
#path to the Python dependents modules.
FILE_PATH='./dependent_modules.txt'

checkModule() {

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
        echo "---- $line module does not exist on current machine! ----"
      fi
  done < "$1"

}

echo '*** Checking if Python exists on the current machine. ***'
if command python3 --version  > /dev/null
then
  echo "++++ Python has been already installed. ++++"
  python3 --version
else
  echo "---- Python does not exist on current machine! ----"
fi

echo '*** Checking if PIP package exists on the current machine. ***'
if command -v pip3 --version > /dev/null
then
  echo "++++ Pip package has been already installed. ++++"
  pip3 --version
  echo '*** Checking solution dependent modules on the current machine. ***'
  #check dependencies
  checkModule "$FILE_PATH"

else
  echo "---- PIP does not exist on current machine! ----"
fi
