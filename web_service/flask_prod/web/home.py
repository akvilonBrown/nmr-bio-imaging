#docker run -d -p 5000:5000 -v /home/iaroslav/docker-volumes/vol:/app nmr-web:1.0
import os
import zipfile
import solutionModule
import cv2
import time
import uuid
import sys
from flask import Flask, render_template, send_file, request, redirect, url_for
from keras.models import load_model

from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime

#from werkzeug import secure_filename
app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = 'files/'
result_fn = 'results.zip'
RESULT_FILE = os.path.join(UPLOAD_FOLDER, result_fn)
INPUT = os.path.join(UPLOAD_FOLDER, 'input')
OUTPUT = os.path.join(UPLOAD_FOLDER, 'output')
NUM_FILES_RETAIN = 5
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['INPROGRESS'] = False
app.config['ML_MODEL'] = os.path.join(UPLOAD_FOLDER, 'model_full_saved.hd')

def empty_zip_folder(path_to_folder):
          flist = os.listdir(path_to_folder)
          count = 0;
          for f in flist:
              if f.endswith('.zip'):
                  count +=1
          
          if (count > NUM_FILES_RETAIN) :
              sys_dir = os.getcwd()
              os.chdir(path_to_folder)
              flist = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime, reverse = True)[NUM_FILES_RETAIN:] 
              os.chdir(sys_dir)
              for f in flist:
                  if f.endswith('.zip'):
                      os.remove(os.path.join(path_to_folder, f))
                      
                  
def empty_folder(path_to_folder):
    for root, dirs, files in os.walk(path_to_folder, topdown=False):
       for name in files:
          print(os.path.join(root, name))
          os.remove(os.path.join(root, name))
       for name in dirs:
          print(os.path.join(root, name))
          os.rmdir(os.path.join(root, name))

@app.route("/", methods=['GET'])
def upload_file():
    return render_template('upld.html')
	
@app.route("/unlock", methods=['GET'])
def unlock():         
    app.config['INPROGRESS'] = False
    return  render_template('info.html', message = 'Session reset')   
    
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file2():
   if request.method == 'POST':
      try: 
          if  not os.path.exists(app.config['ML_MODEL']):           
              return  render_template('info.html', message = 'ML Model is lost on server') 
      
          f = request.files['file']
          if (f.filename == ''):
              return  render_template('info.html', message = 'Empty file. Please upload valid zip archive with images') 
          if not f.filename.endswith('.zip'):
              return  render_template('info.html', message = 'Not a valid archive. Please upload valid zip archive with images')              
          
          saved_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
          f.save(saved_path)
          data_zip = zipfile.ZipFile(saved_path, 'r')
          data_zip.extractall(path=INPUT)
          data_zip.close()
          os.remove(saved_path)
          
          flist = os.listdir(INPUT)
          
          if(len(flist)==0):
              return  render_template('info.html', message = 'Empty archive') 
          
          for f in flist:
              if not(f.endswith('.BMP') or f.endswith('.bmp')) :
                  empty_folder(INPUT)
                  return  render_template('info.html', message = 'Wrong files in archive. Please upload valid zip archive with images') 
                  
          #removing previous result.zip files
          empty_zip_folder(UPLOAD_FOLDER) 
          
          if app.config['INPROGRESS'] :
              return  render_template('info.html', message = 'ML model is busy, somebody is working. Please try later or reset if you are confident this is some mistake')               
          app.config['INPROGRESS'] = True   
          
          return render_template('processing.html')      
      except:          
          return  render_template('info.html', message = 'Unexpected error occured ' + str(sys.exc_info()[0]))  
          
@app.route('/processing2')
def processing2():
        return "OK"
  
@app.route('/processing', methods = ['GET', 'POST'])
def processing():
      solutionModule.runModel(INPUT, OUTPUT, app.config['ML_MODEL'] )
      flist = os.listdir(OUTPUT)
      with zipfile.ZipFile(RESULT_FILE, 'w') as new_zip:
          for name in flist:
              new_zip.write(os.path.join(OUTPUT,name), name)
      new_zip.close()  
      
      empty_folder(INPUT)
      empty_folder(OUTPUT)  
      app.config['INPROGRESS'] = False
      
      return  "Ok" 

# Download form
@app.route("/downloadfile", methods = ['GET'])
def download_file():
    new_resfile = 'result_' + str(uuid.uuid4())  + '.zip'
    dest = os.path.join(UPLOAD_FOLDER, new_resfile)
    os.rename(RESULT_FILE, dest)  
    return render_template('download.html',value=new_resfile)

@app.route('/return-files/<filename>')
def return_files_tut(filename):  
    dest = os.path.join(UPLOAD_FOLDER, filename)    
    return send_file(dest, as_attachment=True, attachment_filename='')

@app.route('/upload_model')
def upload_model():
        return render_template('upldmodel.html')
        
@app.route('/uploader_model', methods = ['GET', 'POST'])
def uploader_model():
   if request.method == 'POST':
      try: 
          f = request.files['file']
          if (f.filename == ''):
              return  render_template('info.html', message = 'Empty request. Please upload valid model')     
          if not (f.filename.endswith('.hd') or f.filename.endswith('.hdf5')):
              return  render_template('info.html', message = 'Not a valid file extention, expected .hd or .hdf5' ) 

          saved_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
          if (saved_path == app.config['ML_MODEL'] ):
              return  render_template('info.html', message = 'Model with such name already exists. New model should have different name') 
          
          f.save(saved_path)
          previous_model = app.config['ML_MODEL'] 
          try:
              model = load_model(saved_path)
          except:
              os.remove(saved_path)
              return render_template('info.html', message = 'Model is not valid')    

          app.config['ML_MODEL'] = saved_path
          os.remove(previous_model)
          
          return  render_template('info.html', message = 'Model successfully updated')    
      except:          
          return  render_template('info.html', message = 'Unexpected error occured ' + str(sys.exc_info()[0]))          
	
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')