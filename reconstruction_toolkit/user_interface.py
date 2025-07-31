# user_interface.py
# ------------------------------------------------------------
# GUI Module for Collecting CT Reconstruction Parameters
#
# This module defines a simple Tkinter-based graphical interface 
# for gathering user inputs required for CT reconstruction:
# - Reader type (Diondo or Nikon)
# - Projection folder and geometry file
# - Reconstruction mode (2D, full 3D, or Z-range)
# - Center of Rotation (COR) options
# - Preprocessing (cleaning, padding)
# - Postprocessing (masking)
# - Output preferences (which results to save)
#
# Outputs:
#   A dictionary of all selected user inputs, to be used in:
#     - load_projections.py
#     - main_reconstruction.py
#
# Dependencies:
#   - tkinter
#   - logger.py (for logging GUI close events)
#
# Author: Kubra Kumrular (RKK)

import os 
from tkinter import Tk, Label, Button, IntVar, Radiobutton, Checkbutton, Entry, Frame, filedialog, messagebox,font, simpledialog

from .logger import write_log 



def get_user_inputs():
    def submit():
        try:
            inputs["reader_type"] = reader_var.get()
            inputs["xml_file"] = xml_file
            if reader_var.get() == 0:
                inputs["proj_folder"] = proj_folder  # 
            else:
                inputs["proj_folder"] = os.path.dirname(xml_file)  # 
            #inputs["proj_folder"] = proj_folder
            inputs["mode"] = mode_var.get()
            inputs["cor_mode"] = cor_var.get()
            inputs["cor_value"] = float(cor_entry.get()) if cor_var.get() == 1 else None
            inputs["z_start"] = int(z_start_entry.get()) if mode_var.get() == 3 else None
            inputs["z_end"] = int(z_end_entry.get()) if mode_var.get() == 3 else None
            inputs["padding"] = padding_var.get()
            inputs["masking"] = masking_var.get()
            inputs["cleaning"] = cleaning_var.get()
            inputs["save_sinogram"] = save_sinogram_var.get()
            inputs["save_central_slice"] = save_central_slice_var.get()
            inputs["save_raw"] = save_raw_var.get()
            root.quit()
            root.destroy()  # close GUI'yi 
        except Exception as e:
            messagebox.showerror("Input Error", str(e))

    root = Tk()
    header_font = ("Arial", 18, "bold")
    section_font = ("Arial", 14)

    root.title("Reconstruction GUI")

    user_closed_gui = {"value": False} 

    def on_close():
        if messagebox.askyesno("Exit Confirmation", "You have unsaved selections. Are you sure you want to exit?"):
            print("User closed the GUI window. Exiting program.")
            write_log("User closed the GUI window. Program terminated.")
            user_closed_gui["value"] = True
            root.quit()

    root.protocol("WM_DELETE_WINDOW", on_close)

    inputs = {}
    xml_file = ""
    proj_folder = ""

    def choose_xml():
        nonlocal xml_file
        if reader_var.get() == 0:
            xml_file = filedialog.askopenfilename(
                title="Select XML File", 
                filetypes=[("XML files", "*.xml")])
        else:
            xml_file = filedialog.askopenfilename(
                title="Select XtekCT File",
                filetypes=[("XTekCT files", "*.xtekct")] )
    
        xml_label.config(text=xml_file)

    def choose_proj():
        nonlocal proj_folder
        proj_folder = filedialog.askdirectory(title="Select Projection Folder")
        proj_label.config(text=proj_folder)

    Label(root, text="0. Select Reader Type", font=header_font).pack(anchor='w')
    reader_var = IntVar(value=0)
    Radiobutton(root, text="Diondo (XML)", variable=reader_var, value=0, font=section_font).pack(anchor='w')
    Radiobutton(root, text="Nikon (XtekCT)", variable=reader_var, value=1, font=section_font).pack(anchor='w')

    Label(root, text="1. Select Geometry Info File", font=header_font).pack(anchor='w')
    Button(root, text="Browse", command=choose_xml, font=section_font).pack(anchor='w')
    xml_label = Label(root, text="No file selected")
    xml_label.pack(anchor='w')

    Label(root, text="2. Select Projection Folder", font=header_font).pack(anchor='w')
    Button(root, text="Browse Folder", command=choose_proj, font=section_font).pack(anchor='w')
    proj_label = Label(root, text="No folder selected")
    proj_label.pack(anchor='w')
    #proj_label = Label(root, text="No folder selected")
    #def update_proj_folder_ui(*args):
    #    if reader_var.get() == 0:
    #        proj_header.pack(anchor='w')
    #        proj_btn.pack(anchor='w')
    #        proj_label.pack(anchor='w')
    #    else:
    #        proj_header.pack_forget()
    ##        proj_btn.pack_forget()
    #        proj_label.pack_forget()

    #reader_var.trace_add("write", update_proj_folder_ui)

    #proj_header = Label(root, text="2. Select Projection Folder", font=header_font)
    #proj_btn = Button(root, text="Browse Folder", command=choose_proj, font=section_font)

    Label(root, text="3. Reconstruction Mode", font=header_font).pack(anchor='w')
    mode_var = IntVar(value=1)
    for text, val in [("1 = Central Slice (2D)", 1), ("2 = Full Volume (3D)", 2), ("3 = Z-range (3D)", 3)]:
        Radiobutton(root, text=text, variable=mode_var, value=val, font=section_font).pack(anchor='w')

    z_frame = Frame(root)
    Label(z_frame, text="Z Start:").pack(side='left')
    z_start_entry = Entry(z_frame, width=5, font=section_font)
    z_start_entry.pack(side='left')
    Label(z_frame, text="Z End:").pack(side='left')
    z_end_entry = Entry(z_frame, width=5, font=section_font)
    z_end_entry.pack(side='left')
    z_frame.pack(anchor='w')

    def toggle_z():
        if mode_var.get() == 3:
            z_frame.pack(anchor='w')
        else:
            z_frame.pack_forget()
    mode_var.trace_add("write", lambda *_: toggle_z())

    Label(root, text="4. COR Mode", font=header_font).pack(anchor='w')
    cor_var = IntVar(value=0)
    Radiobutton(root, text="Auto", variable=cor_var, value=0, font=section_font).pack(anchor='w')
    Radiobutton(root, text="Manual", variable=cor_var, value=1, font=section_font).pack(anchor='w')
    Label(root, text="(Manual COR value should be in **millimeters**)", font=("Arial", 10, "italic"), fg="gray").pack(anchor='w')

    cor_entry = Entry(root, font=section_font)
    cor_entry.pack(anchor='w')

    Label(root, text="5. Cleaning", font=header_font).pack(anchor='w')
    cleaning_var = IntVar(value=0)
    for text, val in [("None", 0), ("Zinger", 1), ("Stripe", 2), ("Both", 3)]:
        Radiobutton(root, text=text, variable=cleaning_var, value=val, font=section_font).pack(anchor='w')

    Label(root, text="6. Padding", font=header_font).pack(anchor='w')
    padding_var = IntVar(value=0)
    for text, val in [("No Padding", 0), ("5%", 1), ("10%", 2)]:
        Radiobutton(root, text=text, variable=padding_var, value=val, font=section_font).pack(anchor='w')

    Label(root, text="7. Masking", font=header_font).pack(anchor='w')
    masking_var = IntVar(value=0)
    for text, val in [("No", 0), ("Yes", 1)]:
        Radiobutton(root, text=text, variable=masking_var, value=val, font=section_font).pack(anchor='w')

    Label(root, text="8. Save Options", font=header_font).pack(anchor='w')
    save_sinogram_var = IntVar(value=1)
    save_central_slice_var = IntVar(value=1)
    save_raw_var = IntVar(value=0)
    Checkbutton(root, text="Save Sinogram", variable=save_sinogram_var, font=section_font).pack(anchor='w')
    Checkbutton(root, text="Save Central Slice", variable=save_central_slice_var, font=section_font).pack(anchor='w')
    Checkbutton(root, text="Save RAW Volume", variable=save_raw_var, font=section_font).pack(anchor='w')

    Button(root, text="Submit", command=submit, font=header_font).pack(pady=10)
    root.mainloop()
    #root.destroy()
    if user_closed_gui["value"]:
        os._exit(0)  
    return inputs
