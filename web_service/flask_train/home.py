import os
import time
import zipfile
#import shutil
from flask import Flask, render_template, send_file, request, redirect, url_for
#from werkzeug import secure_filename
app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = 'files/'
result_fn = 'results.zip'
RESULT_FILE = os.path.join(UPLOAD_FOLDER, result_fn)
INPUT = os.path.join(UPLOAD_FOLDER, 'input')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=['GET'])
def upload_file():
   return render_template('upld.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file2():
   if request.method == 'POST':
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

      flist = os.listdir(INPUT)
      with zipfile.ZipFile(RESULT_FILE, 'w') as new_zip:
          for name in flist:
              new_zip.write(os.path.join(INPUT,name), name)
      #return 'file uploaded successfully'
      #return redirect(url_for('processing'))
      new_zip.close()
      remlist = os.listdir(INPUT)
      for f in remlist:
          os.remove(os.path.join(INPUT, f))
      #return redirect(url_for('download_file', filename = 'results2.zip'))
      #print("uploader before redirect")
      #return redirect(url_for('download_file', filename = 'results.zip'))
      
      #return("Sorry")
      return render_template('processing.html')
      #return redirect(url_for('preprocessing', fname = 'results.zip'))
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
   return("Ok")
   #return redirect(url_for('download_file', filename = 'results.zip')) 

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