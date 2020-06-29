#docker run -d -p 5000:5000 -v /home/iaroslav/docker-volumes/vol:/app nmr-web:1.0
import os
import zipfile
import solutionModule
import cv2
import time
from flask import Flask, render_template, send_file, request, redirect, url_for
#from werkzeug import secure_filename
app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = 'files/'
result_fn = 'results.zip'
RESULT_FILE = os.path.join(UPLOAD_FOLDER, result_fn)
INPUT = os.path.join(UPLOAD_FOLDER, 'input')
OUTPUT = os.path.join(UPLOAD_FOLDER, 'output')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
lockFile = "lock"

@app.route("/", methods=['GET'])
def upload_file():
    return render_template('upld.html')
	
@app.route("/unlock", methods=['GET'])
def unlock():         
    os.remove(lockFile)
    return("Session reset")
    
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file2():
   if request.method == 'POST':
      #if (os.path.exists(lockFile)):
      #  return ("Another session is in progress")
      #open(lockFile, "x")   
      f = request.files['file']
      #f.save(secure_filename(f.filename))
      saved_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
      f.save(saved_path)
      data_zip = zipfile.ZipFile(saved_path, 'r')
      data_zip.extractall(path=INPUT)
      data_zip.close()
      os.remove(saved_path)
      if (os.path.exists(RESULT_FILE)):
          os.remove(RESULT_FILE)
          
      return render_template('processing.html')      
      
@app.route('/processing2')
def processing2():
        return "OK"
  
@app.route('/processing', methods = ['GET', 'POST'])
def processing():
      solutionModule.runModel(INPUT, OUTPUT)
      time.sleep(5)
      flist = os.listdir(OUTPUT)
      with zipfile.ZipFile(RESULT_FILE, 'w') as new_zip:
          for name in flist:
              new_zip.write(os.path.join(OUTPUT,name), name)
      new_zip.close()  
      
      remlist = os.listdir(INPUT)
      for f in remlist:
         os.remove(os.path.join(INPUT, f))
          
      remlist = os.listdir(OUTPUT)
      for f in remlist:
         os.remove(os.path.join(OUTPUT, f))   
      #return redirect(url_for('download_file', filename = 'results2.zip'))
      #if (os.path.exists(lockFile)):
      #  os.remove(lockFile)
      return  "Ok" 

# Download form
@app.route("/downloadfile", methods = ['GET'])
def download_file():
    return render_template('download.html',value=result_fn)

@app.route('/return-files/<filename>')
def return_files_tut(filename):   
    return send_file(RESULT_FILE, as_attachment=True, attachment_filename='')


	
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')