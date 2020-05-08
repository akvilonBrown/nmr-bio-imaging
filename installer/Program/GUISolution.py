import solutionModule
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox
from tkinter import *

class Details:
  def __init__(self):
    self.source_folder = ''
    self.destination_folder = ''

detail_info = Details()

window = Tk()
window.title("Images Segmentation app")
window.geometry('950x200')

slbl = Label(window, text=":")
dlbl = Label(window, text=":")
run_res = Label(window, text="")

slbl.grid(row = 0, column = 1, sticky = W, pady = 2)
dlbl.grid(row = 1, column = 1, sticky = W, pady = 2)
run_res.grid(row = 2, column = 1, sticky = W, pady = 2)

def clickedSource():
    detail_info.source_folder = fd.askdirectory()
    if not detail_info.source_folder :
        messagebox.showinfo('Source folder was not defined!', 'Please select source folder!!')
    else :
        slbl.configure(text="Selected source folder :"+detail_info.source_folder)


def clickedDestination():
    detail_info.destination_folder= fd.askdirectory()
    if not detail_info.destination_folder :
        #dlbl.configure(text="Please select destination folder!! " , background="red", fg="white")
        messagebox.showinfo('Destination folder was not defined!', 'Please select destination folder!!')
    else :
        dlbl.configure(text="Selected destination folder: " + detail_info.destination_folder)

def run_procedure():
    solutionModule.runModel(detail_info.source_folder, detail_info.destination_folder)
    run_res.configure(text="Done")

btn = Button(window, width=23, text="Select source folder!", command=clickedSource)
btn.grid(row = 0, column = 0, sticky = W, pady = 2)
btn1 = Button(window, width=23, text="Select destination folder!", command=clickedDestination)
btn1.grid(row = 1, column = 0, sticky = W, pady = 2)

run_model = Button(window, width=23, padx=5, pady=5, text="Run Segmentation", command=run_procedure)
run_model.grid(row = 2, column = 0, sticky = W, pady = 2)

window.mainloop()
