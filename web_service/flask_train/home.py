import os
import time
import zipfile
import uuid
import sys
#import shutil
from flask import Flask, render_template, send_file, request, redirect, url_for
#from werkzeug import secure_filename
app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = 'files/'
result_fn = 'results.zip'
RESULT_FILE = os.path.join(UPLOAD_FOLDER, result_fn)
INPUT = os.path.join(UPLOAD_FOLDER, 'input')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['INPROGRESS'] = False

def empty_zip_folder(path_to_folder):
          flist = os.listdir(path_to_folder)
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
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file2():
   if request.method == 'POST':
      
      #try:
          f = request.files['file']
          if (f.filename == ''):
              return('Empty file. Please upload valid zip archive with images')
          if not f.filename.endswith('.zip'):
              return('Not a valid archive. Please upload valid zip archive with images') 
         
    
          #f.save(secure_filename(f.filename))
          saved_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
          f.save(saved_path)
          data_zip = zipfile.ZipFile(saved_path, 'r')
          data_zip.extractall(path=INPUT)
          data_zip.close()
          os.remove(saved_path)
          
          flist = os.listdir(INPUT)
          print(flist)
          if(len(flist)==0):
              return('Empty archive')
          
          for f in flist:
              if not(f.endswith('.BMP') or f.endswith('.bmp')) :
                  empty_folder(INPUT)
                  return('Wrong files in archive. Please upload valid zip archive with images')
          
          #removing previous result.zip file
          empty_zip_folder(UPLOAD_FOLDER)     
    
          '''
          if (os.path.exists(RESULT_FILE)):
              os.remove(RESULT_FILE)
          ''' 
          if app.config['INPROGRESS'] :
              return("ML model is busy, somebody is working. Please try later")
          app.config['INPROGRESS'] = True       
          
          res_file = 'result_' + str(uuid.uuid4())+'.zip'
          res_file_path = os.path.join(UPLOAD_FOLDER, res_file)
          flist = os.listdir(INPUT)
          with zipfile.ZipFile(res_file_path, 'w') as new_zip:
              for name in flist:
                  new_zip.write(os.path.join(INPUT,name), name)
          #return 'file uploaded successfully'
          #return redirect(url_for('processing'))
          new_zip.close()
          #time.sleep(5)
          empty_folder(INPUT) 

          #return redirect(url_for('download_file', filename = 'results2.zip'))
          #print("uploader before redirect")
          #return redirect(url_for('download_file', filename = 'results.zip'))
          
          #return("Sorry")
          return render_template('processing.html')
          #return redirect(url_for('preprocessing', fname = 'results.zip'))
      #except:
          #return('Unexpected error happened. ' + str(sys.exc_info()[0]))

@app.route('/unlock', methods = ['GET'])
def unlock():
    app.config['INPROGRESS'] = False
    return("Session unlocked")
'''
@app.route('/preprocessing/<fname>')
def preprocessing(fname):
   
   #return render_template('processing.html') 
   return render_template('processing.html')  
'''    
@app.route('/processing', methods = ['GET', 'POST'])
def processing():
   #return render_template('processing.html') 
   time.sleep(5)
   #return render_template('download.html',value='result.zip')
   #return "Ok"
   print("processing INPROGRESS: " + str(app.config['INPROGRESS']))
   app.config['INPROGRESS'] = False
   return redirect(url_for('download_file', filename = 'results.zip')) 

# Download form
'''
@app.route("/downloadfile/<filename>", methods = ['GET'])
def download_file(filename):
    return render_template('download.html',value=filename)
'''
@app.route("/downloadfile", methods = ['GET'])
def download_file():
    return render_template('download.html',value='result.zip')

@app.route('/return-files/<filename>')
def return_files_tut(filename):
    #file_path = UPLOAD_FOLDER + filename
    return send_file(RESULT_FILE, as_attachment=True, attachment_filename='')


	
if __name__ == '__main__':
   app.run()