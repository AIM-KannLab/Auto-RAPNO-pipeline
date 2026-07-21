"""                                      RAPNO project

This file contains the main function to run the RAPNO project on every datasets and every segmentation models

"""

## IMPORT LIBRARIES

import csv
import os
import pandas as pd
from sentry_sdk import ai
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import shutil
import SimpleITK as sitk
from scipy.spatial.distance import cdist
from sympy import Point
from collections import namedtuple
from skimage.measure import find_contours
import nibabel as nib
from skimage.morphology import label
from tqdm import tqdm
import SimpleITK as sitk
import subprocess
from collections import defaultdict
from datetime import datetime 
from skimage.measure import label

from matplotlib.patches import Patch
from matplotlib.cm import get_cmap
import ast

import torch
torch.cuda.empty_cache()

""" IMPORTANT INFO

We have 3 different csv files to work with:

- 3D volume dataset: it contains 3D volumes and pat_id (OUTPUT OF PIPELINE)

- 2D cross-sectional areas dataset: it contains 2D cross-sectional areas for all planes (axial, sagittal and coronal) or for the only one you specify, diameters, 
                    pat_id and the slice number used to compute 2D cross-sectional areas (OUTPUT OF PIPELINE)

- clinical dataset: it contains the clinical info of patients (trial, pat_id, RT start date, RT end date,
                     Date of First Progression and total_scandates)   (INPUT OF PIPELINE)

Please rename the variables in clinical dataset as specified below:
- trial: name of trial such as PNOC008 or PNOC022 (not necessary)
- pat_id: pat_id
- RT start date: RT_start_date
- RT end date: RT_end_date
- Date of First Progression: Progression_date (first progression after first period of RT)
- total_scandates: total_scandates (the date of scans you have available; available images and masks)
- tumor location: tumor_location (string of explanation from radiologist)
- sex: Sex
- Age:Age

total_scandates presents a list of dates for each patient as follows:
pat_id | total_scandates
1      | [date1 - date2 - date3 - ...]

scandates presented in the list are in the format: yyyymmdd

RT end and start dates are in the format: yyyy-mm-dd

PS: IF YOUR DATASET DOESN'T HAVE THIS FORMAT, PLEASE USE "normalize_columns" FUNCTION TO CHANGE THE NAMES OF THE COLUMNS

We will work with 3 images folders:

- segmented tumor mask (INPUT OF PIPELINE)

- Images (png) with drawing the contours and diameters (OUTPUT OF PIPELINE)

- Images overlapped: multiple semgented tumors mask overlap on a MRI (OUTPUT OF PIPELINE)

"""

"#################################################################################################################################################" 

"""           ########## PREPARE THE CSV DATASET #############   """

"#################################################################################################################################################" 

import re

import re
import pandas as pd

def is_yyyymmdd_format(date_str):
    """Return True if string is in YYYYMMDD format."""
    return bool(re.match(r"^\d{8}$", str(date_str)))


def reformat_date_str(val):
    """Convert a single date string to YYYYMMDD, leaving already-formatted or missing values alone."""
    if pd.isna(val) or str(val).strip() in ("", "N/A"):
        return None
    val = str(val).strip()
    if is_yyyymmdd_format(val):
        return val
    dt = pd.to_datetime(val, errors='coerce')
    return dt.strftime("%Y%m%d") if pd.notna(dt) else None


def reformat_multi_date_str(val):
    """Handle total_scandates fields that may contain multiple dates joined by '-'."""
    if pd.isna(val) or str(val).strip() in ("", "N/A"):
        return None
    parts = [p.strip() for p in str(val).split('-') if p.strip()]
    formatted = [reformat_date_str(p) for p in parts]
    formatted = [f for f in formatted if f is not None]
    return "-".join(formatted) if formatted else None


def normalize_columns(dataset_path, final_dataset_path):
  

    df = pd.read_csv(dataset_path, dtype={"pat_id": str})

    # pat_id is already clean and separate from Trial in this dataset — no split needed.
    if "pat_id" in df.columns:
        df["pat_id"] = df["pat_id"].astype(str).str.strip()

    # RT_start_date / RT_end_date: single date per row
    if "RT_start_date" in df.columns:
        df["RT_start_date"] = df["RT_start_date"].apply(reformat_date_str)

    if "RT_end_date" in df.columns:
        df["RT_end_date"] = df["RT_end_date"].apply(reformat_date_str)

    # total_scandates: may contain multiple dates joined by "-"
    if "total_scandates" in df.columns:
        df["total_scandates"] = df["total_scandates"].apply(reformat_multi_date_str)

    print('this', df.head())

    df.to_csv(final_dataset_path, index=False)

    return df


"#################################################################################################################################################" 

"""            ####          CODE TO OVERLAP MASKS      ####         """

"#################################################################################################################################################" 
"""
This code overlaps the segmentation masks on the MRI images and saves the resulting figures as PNG files in a folder. 

You can decide wich MRI modality you want to use as background (T1w, T2w, FLAIR, etc.) by changing the variable "expected_image_prefix" in the function "save_masks_img".

"""

def _load_mask_dict(scan_list, mask_folder):
    """Load all masks into {scandate: np.ndarray}, including the latest."""
    latest_scan_date, latest_mask, img = scan_list[-1]
    mask_data_dict = {}

    for scandate, mask in scan_list[:-1]:
        mask_path = os.path.join(mask_folder, mask)
        mask_data_dict[scandate] = nib.load(mask_path).get_fdata()

    latest_mask_path = os.path.join(mask_folder, latest_mask)
    mask_data_dict[latest_scan_date] = nib.load(latest_mask_path).get_fdata()

    return mask_data_dict, img


def _extract_slice(volume, plane, slice_idx, orient_fn=None):
    """Extract a 2D slice from a 3D volume given a plane and index."""
    if plane == 'axial':
        slc = volume[:, :, slice_idx]
    elif plane == 'sagittal':
        slc = volume[slice_idx, :, :]
        if orient_fn:
            slc = orient_fn(slc)
    elif plane == 'coronal':
        slc = volume[:, slice_idx, :]
        if orient_fn:
            slc = orient_fn(slc)
    else:
        raise ValueError(f"Unknown plane: '{plane}'")
    return slc




def save_masks_img(mask_folder, image_folder):
    
    pat = {}
    for mask in os.listdir(mask_folder):
        
        if mask.endswith("_mask.nii.gz"):
            basename = mask[:-7]  # remove '.nii.gz'
            id, scan, _ = basename.split("_")

            if id not in pat:
                pat[id]=[(scan, mask)]
            else:
                pat[id].append((scan, mask))
      
    
    #print("pat", pat)
        
    for id in pat:
        pat[id] = sorted(pat[id], key=lambda x: x[0])

    for id, scans in pat.items(): ## we are taking the image of more recent scan; so we can use as backgrund during the overlapping

       # print(id, scans)
        latest_scan_date, latest_mask = scans[-1]

        #print(scans[-1])

        expected_image_prefix = f"{id}_{latest_scan_date}_t1c"   ## CHANGE IF WE WANT DIFFERENT MODALITY AS BACKGROUND
        #print(expected_image_prefix)

        matched_image = None
        for img in os.listdir(image_folder):
            if img.endswith(".nii.gz") and img.startswith(expected_image_prefix):
                matched_image = img
                #print(img)
                break

        
                
        pat[id][-1] = (latest_scan_date, latest_mask, matched_image)  ## replace the last tuple with the new one

        if matched_image is None:
            print(f"[WARNING] No matching image found for patient {id}, "
                  f"scan {latest_scan_date} (expected prefix '{expected_image_prefix}')")
        

        print(pat)

    return pat

def orient_for_display(slice_2d):
    rotated = np.rot90(slice_2d)
   
    return rotated


def overlap_masks_img(pat, mask_folder, img_folder,  output_folder, largest_slice_csv_path = None):

    """
    Overlay segmentation masks on MRI images and save PNG figures.

    With CSV    → pairwise comparison (red = earlier, green = later)
                  using radiologist-selected slice and plane. it's the csv file obtained from the AI-RAPNO pipeline
    Without CSV → one image per scan date, auto-selected axial slice.
    """

    os.makedirs(output_folder, exist_ok=True)

    slice_df = (
        pd.read_csv(largest_slice_csv_path)
        if largest_slice_csv_path is not None
        else pd.DataFrame()
    )

    # Pre-format CSV columns once  
    if not slice_df.empty:
        if 'total_scandates' not in slice_df.columns and 'scandate' in slice_df.columns:
            slice_df = slice_df.rename(columns={'scandate': 'total_scandates'})
        slice_df['pat_id'] = slice_df['pat_id'].astype(str).str.zfill(2)
        slice_df['total_scandates'] = slice_df['total_scandates'].astype(str)

    for patient_id, scan_list in pat.items():

        mask_data_dict, img_file = _load_mask_dict(scan_list, mask_folder)

        img_nii   = nib.load(os.path.join(img_folder, img_file))
        img_data  = img_nii.get_fdata()

        sorted_dates = sorted(mask_data_dict.keys())
        patient_id_str = str(patient_id).zfill(2)
        print(f"Patient {patient_id_str} — dates: {sorted_dates}")

        # ── NO CSV + overlap just mask and 1 img: auto-selected axial slice  ─────────
        if slice_df.empty:
            for scandate in sorted_dates:
                mask = mask_data_dict[scandate]
                _, slice_idx = get_valid_slice(mask, axis=2)

                img_slice  = img_data[:, :, slice_idx]
                mask_slice = mask[:, :, slice_idx]

                out_path = os.path.join(output_folder, f"{patient_id}_{scandate}.png")
                scandate = datetime.strptime(scandate, "%Y%m%d").strftime("%Y-%m-%d")
            

                plt.figure(figsize=(6, 6))
                plt.imshow(img_slice, cmap='gray') 
                plt.imshow(m1_slice, cmap='Reds', alpha=0.6)  ## previous one 
                legend_elements = [
                    Patch(facecolor='red', edgecolor='r', label=f'{scandate_1}'),
                   ]
                plt.legend(handles=legend_elements, fontsize=10, loc='lower right')
                plt.title(f'{scandate_1}', fontsize=14)
                plt.text(20, 20, f'Slice num: {slice_idx}', color='white', fontsize=12)
                plt.axis('off')

                out_name = f"{patient_id}_{scandate}.png"
                out_path = os.path.join(output_folder, out_name)
                plt.savefig(out_path, bbox_inches='tight')
                plt.close()

        # ── WITH CSV + overlap 2 masks and 1 img: pairwise comparison using radiologist slice ─────────────
        else:
            for scandate1, scandate2 in zip(sorted_dates[:-1], sorted_dates[1:]):

                slice_row = slice_df[
                    (slice_df['pat_id'] == patient_id_str) &
                    (slice_df['total_scandates'].str.contains(scandate1))
                ]

                if slice_row.empty:
                    print(f"  No CSV row for {patient_id_str} / {scandate1} — skipping")
                    continue

                row = slice_row.iloc[0]
                slice_idx = int(row['pipeline_slice_number']) ## we are taking the slice number selected by AI-RAPNO pipeline
                plane = row['pipeline_plane'].lower() ## use the plane selected by AI-RAPNO pipeline

                try:
                    img_slice  = _extract_slice(img_data, plane, slice_idx, orient_for_display)
                    m1_slice   = _extract_slice(mask_data_dict[scandate1], plane, slice_idx, orient_for_display)
                    m2_slice   = _extract_slice(mask_data_dict[scandate2], plane, slice_idx, orient_for_display)
                except ValueError as e:
                    print(f"  {e} — skipping {scandate1} vs {scandate2}")
                    continue

                out_path = os.path.join(
                    output_folder, f"{patient_id}_{scandate1}_{scandate2}.png"
                )
               

             
                scandate_1 = datetime.strptime(scandate1, "%Y%m%d").strftime("%Y-%m-%d")
                scandate_2 = datetime.strptime(scandate2, "%Y%m%d").strftime("%Y-%m-%d")

                plt.figure(figsize=(6, 6))
                plt.imshow(img_slice, cmap='gray') 
                plt.imshow(m1_slice, cmap='Reds', alpha=0.6)  ## previous one 
                plt.imshow(m2_slice, cmap='Greens', alpha=0.5)
                legend_elements = [
                    Patch(facecolor='red', edgecolor='r', label=f'{scandate_1}'),
                    Patch(facecolor='green', edgecolor='g', label=f'{scandate_2}')
                    ]
                plt.legend(handles=legend_elements, fontsize=10, loc='lower right')
                plt.title(f'ID {patient_id}: {scandate_1} vs {scandate_2}', fontsize=14)
                plt.text(20, 20, f'Slice num: {slice_idx}', color='white', fontsize=12)
                plt.axis('off')

                out_name = f"{patient_id}_{scandate1}_{scandate2}.png"
                out_path = os.path.join(output_folder, out_name)
                plt.savefig(out_path, bbox_inches='tight')
                plt.close()
                print(f"Saved overlay: {out_path}")


        


"#################################################################################################################################################"    

"""         ####         MAIN FUNCTIONS TO COMPUTE 2D CROSS SECTIONAL AREA     ####         """

"#################################################################################################################################################" 

## Eclidean /pairwise distance

class Point(namedtuple('Point', 'x y')):
	__slots__ = ()
	@property
	def length(self):
		return (self.x ** 2 + self.y ** 2) ** 0.5 #length from the origin
	def __sub__(self, p):
		return Point(self.x - p.x, self.y - p.y) #subtract self.x, self.y by coordinates of Point p
	def __str__(self):
		return 'Point: x=%6.3f  y=%6.3f  length=%6.3f' % (self.x, self.y, self.length)
		#print the coordinates and the length of the Point

	
		


def compute_pairwise_distances(P1, P2, min_length=10): ## change and CHECK D!!!
    """
    Compute pairwise Euclidean distances between points in P1 and P2, 
    filtering out pairs with distances below `min_length`.

    Output is the pairwise matrix and a sorted list of distances
    """
	#creates a 2D array where the element correlate with the location between the point from P1 and point from P2
    # Compute Euclidean distance matrix
    euc_dist_matrix = cdist(P1, P2, metric='euclidean')
    #print(euc_dist_matrix)
    

    indices = []
    for x in range(euc_dist_matrix.shape[0]):
        for y in range(euc_dist_matrix.shape[1]): 
            p1 = Point(*P1[x])
            p2 = Point(*P2[y])
            d = euc_dist_matrix[x, y]

            # Skip if points are the same or below minimum distance threshold
            if p1 != p2 or d > min_length:
                #print(p1,p2,d)
                indices.append([p1, p2, d])  ## d is in mm (as voxel is in mm)

    # Sort valid indices by increasing distance
    sorted_indices = sorted(indices, key=lambda x: x[2], reverse = True)
    #print("Euclidean Distance Matrix:")
	#print(euc_dist_matrix)
	#print("\nSorted Pairs (Point1, Point2, Distance):")
    
    for p1, p2, dist in sorted_indices:
        printed = '({p1}, {p2}) -> Distance: {dist}'
       # print(f"({p1}, {p2}) -> Distance: {dist}")

    return euc_dist_matrix, sorted_indices
    

   

## Take the maximum distances and figure out if they are perpendicular

def interpolate(p1, p2, d):
  "Interpolate-> create the lines creating a lot of points between p1 and p2"
	#numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
  if not np.isfinite(d) or d <= 0:
     # print(f"Skipping interpolation: invalid distance {d} for points ({p1}, {p2})")
      return []  # Return empty list if distance is invalid
  
  X = np.linspace(p1.x, p2.x, int(round(d))).astype(int)
  Y = np.linspace(p1.y, p2.y, int(round(d))).astype(int)
    
    # Create unique points from X and Y coordinates
  XY = np.asarray(list(set(zip(X, Y))))
    
  return XY


def ccw(A,B,C):
    "Counterwise order test--> to see if there is perpendicularity"
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)


def intersect(A,B,C,D):
    "Check the intersection of 2 lines"
    
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def vector_norm(p):
    "The length of vector"
    length = p.length  # Ensure this attribute exists in `p`
    
    if length == 0:
        return Point(0, 0)  # Handle division by zero
    else:
        return Point(p.x / length, p.y / length)  # Properly indented return



def max_distance(sorted_indices,  img, tolerance):  
    #print(sorted_indices)

    for i, (p1, p2, d1) in enumerate(sorted_indices):
     # print("first diam", p2,p1,d1)
      XY = interpolate(p1, p2, d1)
     # print(XY)
      intersections = sum(img[x, y] == 0 for x, y in XY)  ## to see if we are considering background!

     
      intersections_ratio = intersections / float(len(XY))
      
      if intersections_ratio < 0.1: 
        V = vector_norm(Point(p2.x - p1.x, p2.y - p1.y)) ## length =1 because normalized
        #print("v", V)

        #print("check first pairs")
        
        for j, (q1, q2, d2) in enumerate(sorted_indices[i:]):
          dist_p1_q1 = np.sqrt((q1.x - p1.x) ** 2 + (q1.y - p1.y) ** 2)  # Calculate distance between p1 and q1
          if dist_p1_q1 < 10:  # Skip if points are too close
          #  print("point to close!!")
            continue
          W = vector_norm(Point(q2.x - q1.x, q2.y - q1.y))
        #  print("W", W)


          if abs(np.dot(V, W)) < tolerance:
            XY = interpolate(q1, q2, d2)
            intersections = sum(img[x, y] == 0 for x, y in XY)

          
            intersections_ratio = intersections / float(len(XY))

         # print("check second pairs")
             
            if intersections_ratio < 0.1 and intersect(p1, p2, q1, q2): ##
            #max_perpendicular_pair = (p1, p2, q1, q2)
            # 
               #print(f"Perpendicular Pair Found: {p1}, {p2}, {q1}, {q2}")  # print different points so good

            
               return p1, p2, q1, q2



def get_neighbors(mask):
    "Connectivity and filter points"
    conn_comp = label(np.round(mask),connectivity=3).astype(float)
  #  print("connected", conn_comp)
    ## Empty and new mask
    new_mask = np.zeros_like(mask)

    # Loop through each connected component
    for i in range(1,len(np.unique(conn_comp))):  # Labels start from 1
        # Create a mask for the current component
        curr_mask = np.zeros(conn_comp.shape)
        idx = np.where(conn_comp==np.unique(conn_comp)[i]) #index where the connected component mask equals the component of interest
        curr_mask[idx] = 1.0
        curr_mask = curr_mask.astype(int)
       # print(curr_mask)

        if curr_mask is None:
           print('None',curr_mask)
       
        new_mask[curr_mask == 1] = 1  # Mark the selected component as 1 in the new mask
    

    return new_mask   ##3D mask




def rapno_2D_area(max_perpendicular_pair, vox_x=1):
    "Compute diameters and the final RAPNO measure"

    p1,p2,q1,q2 = max_perpendicular_pair   ## these are 4 points

    rapno_measure = ((p2 - p1).length * (q2 - q1).length) * vox_x * vox_x

    diam2= round((q2 - q1).length, 2)
    diam1 =  round((p2 - p1).length, 2)

    return rapno_measure, diam1, diam2




def plot_contours(contours, lw=4, alpha=0.5):
    "For contouring visualization"
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=lw, alpha=alpha, c='r')
          



def get_valid_slice(mask, axis):   ## 0 for sagittal, 1 for coronal and 2 for axial
    "## Select the slice for each slide. Images can have tumors at different slide."
    "Take the slice with maximum number of pixel =1 (maximum area) "
  
# Order slices by descending lesion area (GIVE ME THE SLICE NUMBER OF EACH AREA)
    area_sorted_ix = list(np.sum(mask, axis=(0, 1)).argsort()[::-1]) if axis == 2 else \
                     list(np.sum(mask, axis=(0, 2)).argsort()[::-1]) if axis == 1 else \
                     list(np.sum(mask, axis=(1, 2)).argsort()[::-1])
    
    max_area = -1
    for j in area_sorted_ix:
        slice = mask.take(j, axis = axis)
        area = np.sum(slice)

        if area > max_area:
            max_area = area

            max_slice = slice
          #  print(type(max_slice), np.sum(slice), j)
    
            return max_slice, j
    



def plot_contour_pixels(image, contours):
    plt.imshow(image, cmap="gray")
    
    
    for point in contours:
      y, x = int(point[0]), int(point[1])  # Get pixel coordinates
      plt.text(x, y, str(image[y, x]), color="red", fontsize=8, ha="center", va="center")

    plt.title("Contour Area with Pixel Values")
    plt.axis("off")
   # plt.show()




def process_contours(plane_spec, slice_mask, plane, file, scandate, id, results_dict, vox_x=1, target_slice = None, current_slice = None):
    
    """Process contours and calculate diameters for a given plane."""
    if slice_mask is None or np.all(slice_mask == 0):
        #print(f"Skipping {file}: No segmentation found in {plane} plane.")
        return
    
    slice_mask[slice_mask == 2] = 0 ## edema ot count!!!
    slice_mask[slice_mask == 1] = 255
    slice_mask[slice_mask == 3] = 255




    
    labeled_mask = label(np.round(slice_mask>0), connectivity=2).astype(float) ## LABEL FEATURES IN A IMAGE!!!
    num_lesions = labeled_mask.max()
    #print(labeled_mask, num_lesions)

    color_map = get_cmap("tab10")
    fig = plt.figure(figsize=(10, 10), frameon=False)
    plt.margins(0, 0)
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    background_image = None
    if background_image is not None:
        plt.imshow(background_image, cmap='gray')
    else:
        plt.imshow(slice_mask, cmap='gray')

    results = []

    text_y = 10
    for  lesion_id in range(1, int(num_lesions) + 1):
        # Step 2: Extract the binary mask for this specific lesion
        lesion_mask = (labeled_mask == lesion_id).astype(np.uint8) * 255
      #  print(lesion_mask)



        contours = find_contours(lesion_mask, level=1)  ## find the contours of the mask at each slide
       # print("contours", contours)


        if len(contours) == 0:
        #print(f"No {plane} contours found for {file}")
           return

    #comb_contours = contours[0]
    ##for i in range(1,len(contours)):
      #  comb_contours = np.concatenate((comb_contours,contours[i]))
    
    #combined_contours = comb_contours.astype(int)
    
        combined_contours = np.concatenate(contours).astype(int)
    #print(combined_contours)
    
    
        euc_dist_matrix, ordered_diameters = compute_pairwise_distances(combined_contours, combined_contours) ## compute the pairwise matrix
    #print(type(ordered_diameters))
        result = max_distance(ordered_diameters, slice_mask, tolerance=0.1)
        print("result for 1 lesion", result)

        if result is None:
        #print("Error: max_distance returned None. Unable to unpack values.")
    # Handle the error case (e.g., continue to the next iteration, return early, or set default values)
            p1, p2, q1, q2 = (None, None, None, None)  # or any other appropriate default values
            rapno_measure = None
            diam1 = None
            diam2 = None
            results.append(("-", "-", "-"))

            continue  
        
        else:
            p1, p2, q1, q2 = result
            diam1 = (p2 - p1).length
            diam2 = (q2 - q1).length

            diam1_cm = diam1/10
            diam2_cm = diam2/10  # convert mm → cm
            
            rapno_measure = diam1_cm * diam2_cm

            if diam1_cm < 1 or diam2_cm < 1:  ## NOT MEASURABLE 
                print("One of the 2 diamters is less than 1 cm, skipping this lesion.")
                continue
        #print("rapno measure", rapno_measure)
    #rapno_measure, diam1, diam2 = rapno_2D_area(max_perp_points)
   
            print("for 1 lesion", diam1_cm, diam2_cm, rapno_measure)
            results.append((round(rapno_measure, 2), diam1_cm, diam2_cm))

        ### Save the mask with diameters and contours   #####

    
                    
            color_d1 = color_map((2 * lesion_id - 1) % 10)
            color_d2 = color_map((2 * lesion_id) % 10)

    # Plot contours and diameters
            plot_contours(contours, lw=1.5, alpha=1.0)
            D1 = np.asarray([[p1.x, p2.x], [p1.y, p2.y]])
            D2 = np.asarray([[q1.x, q2.x], [q1.y, q2.y]])
    
            plt.plot(D1[1, :], D1[0, :], lw=2, c=color_d1, label=f'Lesion {lesion_id} D1: {round(diam1_cm, 2)} cm')
            plt.plot(D2[1, :], D2[0, :], lw=2, c=color_d2, label=f'Lesion {lesion_id} D2: {round(diam2_cm, 2)} cm')

    # Dynamically position text
            plt.text(10, text_y, f'Lesion {lesion_id} RAPNO: {round(rapno_measure, 2)} cm²', fontsize=12, color='r')
            text_y += 7
        
        

 #   if  target_slice is not None:
  #      if current_slice != target_slice:
  #          plt.close()
   #         return results

    filename = file[:-7]

    output_file = f'{img_folder}/{filename}_{plane_spec}_{current_slice}.png'
    plt.legend(fontsize = 16)
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0, dpi=100)
    plt.close(fig)

            
    print("results", results)
        
        
    return results




"#################################################################################################################################################" 

"""   ####   FUNCTION TO CREATE A CSV WITH LARGEST PERPENDICULAR DIAMETERS (D1, D2) AND AREA FOR EACH SLICE FOR EACH SCAN #### """

"#################################################################################################################################################" 
"""
It saves the cross-sectional measurments for each slice of each scan per each plane (Sagittal, Coronal, Axial) in a CSV file. 
Then It computes the largest perpendicular diameters and area among all scans and plane and saves them in a separate CSV file.

The final csv file has the lagest perpendicular diameters and area among slices and plane for each scan each scan. 
"""
def compute_diameters_all_slices(plane_spec, mask, axis, file, scandate, id, vox_x=1):
    """        
    Returns:
        dict of {slice_index: [(area_cm2, diameter1_cm, diameter2_cm), ...]} for each lesion
    """
    from collections import defaultdict

    results_dict = []
    num_slices = mask.shape[axis]

    filename_key = file[:-12]  # match your key format
    print(filename_key)
   

    for slice_idx in range(num_slices):
      #  print("slices", filename_key, slice_idx, target_slice)
       # if slice_idx != target_slice:        ## WORK ONLY ON SLIDES YOU WANT!!
       #     continue 

        slice_mask = np.take(mask, slice_idx, axis=axis)

        if np.sum(slice_mask) == 0:
            continue      ## Skip empty slices
        
      #  res = process_contours(slice_mask, plane, file, scandate, id, results_dict, vox_x=vox_x)
        res = process_contours(plane_spec, slice_mask, plane, file, scandate, id, results_dict, vox_x=vox_x, target_slice = None, current_slice=slice_idx)
        print("res", res)
        if len(res) > 1:
            # Check if any lesion is '-'
            if any(t[0] == '-' for t in res):
                continue
            else:
                # SAVE MULTIPLE AREAS all rapno_measure values safely
                multiple_rapno_measure = " and ".join(str(t[0]) for t in res)

                diam1_values = [str(t[1]) for t in res]  # collect diam1s as strings
                diam1_list = " and ".join(diam1_values)
               
                diam2_values = [str(t[2]) for t in res]  # collect diam1s as strings
                diam2_list = " and ".join(diam2_values)

                results_dict.append((multiple_rapno_measure, diam1_list, diam2_list, slice_idx))

        elif len(res) == 1:
            if res[0][0] != '-':
                rapno_measure = float(res[0][0])
                diam1 = float(res[0][1])
                diam2 = float(res[0][2])

                results_dict.append((rapno_measure, diam1, diam2, slice_idx))

    return results_dict



def tumor_measurements_2d_all_slices(plane, folder_tumor, output_folder_2D, data_path):
   
   ##in case of no dataset skip these 2 lines
  # if data_path is not None:
   # data = pd.read_csv(data_path)
  #  total_scandates = sorted(data["total_scandates"].tolist())  # Ensure chronological order
  

    default_keys = {
    "Axial_area": [],
    "Axial_d1": [],
    "Axial_d2": [],
    "Slice_number_Axial": [],
    "Sagittal_area": [],
    "Sagittal_d1": [],
    "Sagittal_d2": [],
    "Slice_number_Sagittal": [],
    "Coronal_area": [],
    "Coronal_d1": [],
    "Coronal_d2": [],
    "Slice_number_Coronal": [],
    "Number_of_lesions": []}

    results_dict = defaultdict(lambda: defaultdict(lambda: {k: [] for k in default_keys}))
   
    
    # Loop over all files in the tumor folder
    for file in os.listdir(folder_tumor):
      #  print(file)
        if file.endswith("_mask.nii.gz"):  ## CHANGE BASED ON HOW YOU CALL YOUR PREDICTIONS
            filename = file[:-7]
           # id, scandate = filename.split('_')  # CHANGE THIS PART OF CODE IF YOU HAVE ADDED AN ADDITIONAL PART TO THE NAME
          #  trial, id, scandate, _ = filename.split('_')
            id, scandate, _ = filename.split('_')
         #   print( id, scandate)

    

         #   id_full = f"{trial}_{id}" ## did it because i have the name of clinical trial

        #   results_dict[id]["trial"] = trial
         #   results_dict[id]["id_full"] = id_full
           
            
            image_path = os.path.join(folder_tumor, file)
            img_data = nib.load(image_path).get_fdata()
        
            new_mask = get_neighbors(img_data)  # Reduce the number of points of mask
            #print("binary mask", np.unique(new_mask))


            # Process the different planes
            if plane == "Axial":
                slice_results = compute_diameters_all_slices(plane, new_mask, 2, file, scandate, id, vox_x=1)
                
                if slice_results and len(slice_results) > 0:
                    for slices in slice_results:
                        rapno_measure, diam1, diam2, num_slice = slices
                        print("slice", slices)
            
            # Append diameters if valid, else "-"
                        if diam1 is not None and diam2 is not None:
                            if isinstance(diam1, (int, float)) and isinstance(diam2, (int, float)):
                                results_dict[id][scandate]["Axial_d1"].append(round(diam1, 2))
                                results_dict[id][scandate]["Axial_d2"].append(round(diam2, 2))
                                results_dict[id][scandate]["Slice_number_Axial"].append(num_slice)

                            elif isinstance(diam1, str) and isinstance(diam2, str):
                                results_dict[id][scandate]["Axial_d1"].append(diam1)
                                results_dict[id][scandate]["Axial_d2"].append(diam2)
                                results_dict[id][scandate]["Slice_number_Axial"].append(num_slice)
                            else:
                                results_dict[id][scandate]["Axial_d1"].append("-")
                                results_dict[id][scandate]["Axial_d2"].append("-")
                                results_dict[id][scandate]["Slice_number_Axial"].append("-")
                        else:
                            results_dict[id][scandate]["Axial_d1"].append("-")
                            results_dict[id][scandate]["Axial_d2"].append("-")
                            results_dict[id][scandate]["Slice_number_Axial"].append("-")

                            # Save total area sum separately if needed
                        results_dict[id][scandate]["Axial_area"].append(rapno_measure)
                    
                else:
                    results_dict[id][scandate]["Axial_area"].append("-")
                    results_dict[id][scandate]["Axial_d1"].append("-")
                    results_dict[id][scandate]["Axial_d2"].append("-")
                    results_dict[id][scandate]["Slice_number_Axial"].append("-")
                    results_dict[id][scandate]["Axial_area"] = "-"

                
            elif plane == "Sagittal":
                slice_results = compute_diameters_all_slices(plane, new_mask, 0, file, scandate, id, vox_x=1)
                
                if slice_results and len(slice_results) > 0:
                    for slices in slice_results:
                        rapno_measure, diam1, diam2, num_slice = slices
            
                        if diam1 is not None and diam2 is not None:
                            if isinstance(diam1, (int, float)) and isinstance(diam2, (int, float)):
                                results_dict[id][scandate]["Sagittal_d1"].append(round(diam1, 2))
                                results_dict[id][scandate]["Sagittal_d2"].append(round(diam2, 2))
                                results_dict[id][scandate]["Slice_number_Sagittal"].append(num_slice)
                            elif isinstance(diam1, str) and isinstance(diam2, str):
                                results_dict[id][scandate]["Sagittal_d1"].append(diam1)
                                results_dict[id][scandate]["Sagittal_d2"].append(diam2)
                                results_dict[id][scandate]["Slice_number_Sagittal"].append(num_slice)
                        
                            else:
                                results_dict[id][scandate]["Sagittal_d1"].append("-")
                                results_dict[id][scandate]["Sagittal_d2"].append("-")
                                results_dict[id][scandate]["Slice_number_Sagittal"].append("-")
                        else:
                            results_dict[id][scandate]["Sagittal_d1"].append("-")
                            results_dict[id][scandate]["Sagittal_d2"].append("-")
                            results_dict[id][scandate]["Slice_number_Sagittal"].append("-")

                            # Save total area sum separately if needed
                        results_dict[id][scandate]["Sagittal_area"].append(rapno_measure)
                    
                else:
                    results_dict[id][scandate]["Sagittal_area"].append("-")
                    results_dict[id][scandate]["Sagittal_d1"].append("-")
                    results_dict[id][scandate]["Sagittal_d2"].append("-")
                    results_dict[id][scandate]["Slice_number_Sagittal"].append("-")
                    results_dict[id][scandate]["Sagittal_area"] = "-"

            elif plane == "Coronal":
                slice_results = compute_diameters_all_slices(plane, new_mask, 1, file, scandate, id, vox_x=1)
                
                if slice_results and len(slice_results) > 0:
                    for slice in slice_results:
                        rapno_measure, diam1, diam2, num_slice = slices
    
        
                        if diam1 is not None and diam2 is not None:
                            if isinstance(diam1, (int, float)) and isinstance(diam2, (int, float)):
                                results_dict[id][scandate]["Coronal_d1"].append(round(diam1, 2))
                                results_dict[id][scandate]["Coronal_d2"].append(round(diam2, 2))
                                results_dict[id][scandate]["Slice_number_Coronal"].append(num_slice)

                            elif isinstance(diam1, str) and isinstance(diam2, str):
                                results_dict[id][scandate]["Coronal_d1"].append(diam1)
                                results_dict[id][scandate]["Coronal_d2"].append(diam2)
                                results_dict[id][scandate]["Slice_number_Coronal"].append(num_slice)
                            else:
                                results_dict[id][scandate]["Coronal_d1"].append("-")
                                results_dict[id][scandate]["Coronal_d2"].append("-")
                                results_dict[id][scandate]["Slice_number_Coronal"].append("-")
                        else:
                            results_dict[id][scandate]["Coronal_d1"].append("-")
                            results_dict[id][scandate]["Coronal_d2"].append("-")
                            results_dict[id][scandate]["Slice_number_Coronal"].append("-")

                            # Save total area sum separately if needed
                        results_dict[id][scandate]["Coronal_area"].append(rapno_measure)
                    
                else:
                    results_dict[id][scandate]["Coronal_area"].append("-")
                    results_dict[id][scandate]["Coronal_d1"].append("-")
                    results_dict[id][scandate]["Coronal_d2"].append("-")
                    results_dict[id][scandate]["Slice_number_Coronal"].append("-")
                    results_dict[id][scandate]["Coronal_area"] = "-"

            elif plane == "all":
                #print("Axial", slice_mask_axial, num_slice_axial, "Coronal", slice_mask_coronal, num_slice_coronal, "Sagittal", slice_mask_sagittal, num_slice_sagittal)
                planes = {
                    "Axial": (new_mask, 2),
                    "Sagittal": (new_mask, 0),
                    "Coronal": (new_mask, 1)
                }
                for p, (mask, axis) in planes.items():
                    slice_results = compute_diameters_all_slices(p, mask, axis, file, scandate, id, vox_x=1)
                #    print("result", result)
                    #if result is None or not None:
                    if slice_results and len(slice_results) > 0:
                        for slice in slice_results:
                            rapno_measure, diam1, diam2, num_slice = slice

                            if p == "Axial":
                                results_dict[id][scandate]["Axial_area"].append(rapno_measure)
                                if diam1 is not None and diam2 is not None:
                                    if isinstance(diam1, (int, float)) and isinstance(diam2, (int, float)):
                                        results_dict[id][scandate]["Axial_d1"].append(round(diam1, 2))
                                        results_dict[id][scandate]["Axial_d2"].append(round(diam2, 2))
                                        results_dict[id][scandate]["Slice_number_Axial"].append(num_slice)

                                    elif isinstance(diam1, str) and isinstance(diam2, str):
                                        results_dict[id][scandate]["Axial_d1"].append(diam1)
                                        results_dict[id][scandate]["Axial_d2"].append(diam2)
                                        results_dict[id][scandate]["Slice_number_Axial"].append(num_slice)

                                    else:
                                        results_dict[id][scandate]["Axial_d1"].append("-")
                                        results_dict[id][scandate]["Axial_d2"].append("-")
                                        results_dict[id][scandate]["Slice_number_Axial"].append("-")
                                else:
                                    results_dict[id][scandate]["Axial_d1"].append("-")
                                    results_dict[id][scandate]["Axial_d2"].append("-")
                                    results_dict[id][scandate]["Slice_number_Axial"].append("-")

                            elif p == "Sagittal":
                                results_dict[id][scandate]["Sagittal_area"].append(rapno_measure)

                                if diam1 is not None and diam2 is not None:
                                    if isinstance(diam1, (int, float)) and isinstance(diam2, (int, float)):
                                        results_dict[id][scandate]["Sagittal_d1"].append(round(diam1,2))
                                        results_dict[id][scandate]["Sagittal_d2"].append(round(diam2,2))
                                        results_dict[id][scandate]["Slice_number_Sagittal"].append(num_slice)

                                    elif isinstance(diam1, str) and isinstance(diam2, str):
                                        results_dict[id][scandate]["Sagittal_d1"].append(diam1)
                                        results_dict[id][scandate]["Sagittal_d2"].append(diam2)
                                        results_dict[id][scandate]["Slice_number_Sagittal"].append(num_slice)
                            
                                    else:
                                        results_dict[id][scandate]["Sagittal_d1"].append("-")
                                        results_dict[id][scandate]["Sagittal_d2"].append("-")
                                        results_dict[id][scandate]["Slice_number_Sagittal"].append("-")
                                else:
                                    results_dict[id][scandate]["Sagittal_d1"].append("-")
                                    results_dict[id][scandate]["Sagittal_d2"].append("-")
                                    results_dict[id][scandate]["Slice_number_Sagittal"].append("-")
                                
                            elif p == "Coronal":
                                results_dict[id][scandate]["Coronal_area"].append(rapno_measure)  
                                if diam1 is not None and diam2 is not None:
                                    if isinstance(diam1, (int, float)) and isinstance(diam2, (int, float)):
                                        results_dict[id][scandate]["Coronal_d1"].append(round(diam1,2))
                                        results_dict[id][scandate]["Coronal_d2"].append(round(diam2,2))
                                        results_dict[id][scandate]["Slice_number_Coronal"].append(num_slice)

                                    elif isinstance(diam1, str) and isinstance(diam2, str):
                                        results_dict[id][scandate]["Coronal_d1"].append(diam1)
                                        results_dict[id][scandate]["Coronal_d2"].append(diam2)
                                        results_dict[id][scandate]["Slice_number_Coronal"].append(num_slice)    
                                    else:
                                        results_dict[id][scandate]["Coronal_d1"].append("-")
                                        results_dict[id][scandate]["Coronal_d2"].append("-")
                                        results_dict[id][scandate]["Slice_number_Coronal"].append("-")
                                else:
                                    results_dict[id][scandate]["Coronal_d1"].append("-")
                                    results_dict[id][scandate]["Coronal_d2"].append("-")
                                    results_dict[id][scandate]["Slice_number_Coronal"].append("-")

            else:
                raise ValueError("Invalid plane specified. Choose from 'Axial', 'Sagittal', 'Coronal', or 'all'.")

    # Prepare the results for saving into a CSV
    results = []
    print("resuts_dict done")
    for id, scan_data in results_dict.items():

        id = str(id).zfill(2)
       # trial = scan_data.get("trial", "")
       # id_full = scan_data.get("id_full", "")
      
        print("results_dict", results_dict)
    
    # Sort scandates chronologically (assumes sortable format like YYYYMMDD or YYYY-MM-DD)
        sorted_scandates = sorted(scan_data.keys())
       # sorted_scandates = sorted([k for k in scan_data.keys() if k not in ("trial", "id_full")])
    
        axial_areas = []
        axial_d1 = []
        axial_d2 = []
        axial_slices = []
    
        sagittal_areas = []
        sagittal_d1 = []
        sagittal_d2 = []
        sagittal_slices = []
    
        coronal_areas = []
        coronal_d1 = []
        coronal_d2 = []
        coronal_slices = []

        for scandate in sorted_scandates:
            data = scan_data[scandate]
        
            axial_areas = data.get("Axial_area", []) if isinstance(data.get("Axial_area"), list) else [data.get("Axial_area")]
            axial_d1 = data.get("Axial_d1", [])
            axial_d2 = data.get("Axial_d2", [])
            axial_slices = data.get("Slice_number_Axial", [])

            sagittal_areas = data.get("Sagittal_area", []) if isinstance(data.get("Sagittal_area"), list) else [data.get("Sagittal_area")]
            sagittal_d1 = data.get("Sagittal_d1", [])
            sagittal_d2 = data.get("Sagittal_d2", [])
            sagittal_slices = data.get("Slice_number_Sagittal", [])

            coronal_areas = data.get("Coronal_area", []) if isinstance(data.get("Coronal_area"), list) else [data.get("Coronal_area")]
            coronal_d1 = data.get("Coronal_d1", [])
            coronal_d2 = data.get("Coronal_d2", [])
            coronal_slices = data.get("Slice_number_Coronal", [])

            results.append([ 
                id,
                scandate,
                ",".join(map(str, axial_areas)),
                ",".join(map(str, axial_d1)),
                ",".join(map(str, axial_d2)),
                ",".join(map(str, axial_slices)),
                ",".join(map(str, sagittal_areas)),
                ",".join(map(str, sagittal_d1)),
                ",".join(map(str, sagittal_d2)),
                ",".join(map(str, sagittal_slices)),
                ",".join(map(str, coronal_areas)),
                ",".join(map(str, coronal_d1)),
                ",".join(map(str, coronal_d2)),
                ",".join(map(str, coronal_slices)),
            ])

    print(results)

    columns = [
        "pat_id", "total_scandates",
        "Axial_Area_(cm2)", "Axial_D1_(cm2)","Axial_D2_(cm2)", "Slice_number_Axial", 
        "Sagittal_Area_(cm2)", "Sagittal_D1_(cm2)","Sagittal_D2_(cm2)", "Slice_number_Sagittal",
        "Coronal_Area_(cm2)", "Coronal_D1_(cm2)", "Coronal_D2_(cm2)", "Slice_number_Coronal"
    ]
    
    area2D = pd.DataFrame(results, columns=columns)
    output_path = os.path.join(output_folder_2D, f"cross_sectional_2D_all_slices.csv")
    area2D.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    return area2D


    
      
def slice_value(variable):
    parts = str(variable).split(',')
    values = []

    for part in parts:
        part = part.strip()
        try:
            values.append(int(float(part)))  # convert to int for slice number
        except ValueError:
            values.append(None)  # mark as missing
    return values

def float_value(variable):  ## take the maximum among slices from the same plane; CASES WITH 2 LESIONS-> CONSIDER SEPARATELY
    parts = str(variable).split(',')
 #   print(parts)
    values = []
    
    for part in parts:
        subparts = part.split('and')  # then split by "and"
        for sub in subparts:
            sub = sub.strip()
            if not sub:
                continue
            try:
                values.append(round(float(sub), 2))  # round to 2 decimals if needed
            except ValueError:
                continue 

    return values

def float_value_no_and(variable):  ## SUM AREAS WHEN DOUBLE DIAMTERS

    parts = str(variable).split(',')
 #   print(parts)
    values = []
    
    for part in parts:
        if "and" not in part:
            part = part.strip()
            if not part:
                continue
    
            values.append(round(float(part), 2))
        else:
            values.append(part)

    return values
      
      
def compute_largest_diam_and_area(row):
    plane_names = ['axial', 'sagittal', 'coronal']
    diam_data = {}
    area_data = {}

    max_diam = -1
    max_plane = None
    max_index = None
    max_slice = -1
    max_area = None
    lesions_2 = "no"

    for plane in plane_names:
        d1_col = f'{plane.capitalize()}_D1_(cm2)'
        d2_col = f'{plane.capitalize()}_D2_(cm2)'
        area_col = f'{plane.capitalize()}_Area_(cm2)'
        slice_col = f'Slice_number_{plane.capitalize()}'

        d1_values = float_value(row[d1_col])
        d1_values_and_cases = float_value_no_and(row[d1_col]) ## only for not splitting the and cases
      #  print("d1 values", d1_values)
        d2_values = float_value(row[d2_col])
        d2_values_and_cases = float_value_no_and(row[d2_col]) ## only for not splitting the and cases
      #  print("d1 values", d2_values)
        area_values = float_value_no_and(row[area_col])
      #  print("areas", area_values)
        slice_values = slice_value(row[slice_col])

        max_len = max(len(d1_values), len(d2_values),  len(slice_values))
      #  print(len(d1_values), len(d2_values), len(slice_values)) ## same value 
        diam_slices = []

        for i in range(max_len):
            d1 = d1_values[i] if i < len(d1_values) else None
            d2 = d2_values[i] if i < len(d2_values) else None
          #  area = area_values[i] if i < len(area_values) else None
            slice = slice_values[i] if i < len(slice_values) else None
        #    print("case", d1, d2, slice, area)

            if d1 is not None and d2 is not None:
                diam = max(d1, d2)
            elif d1 is not None:
                diam = d1
            elif d2 is not None:
                diam = d2
            else:
                diam = None
       #     print("diam max", diam)
            diam_slices.append((diam, slice))  ## maximum between d1 and d2 + area
        
        
    #    print("res", diam_slices)

        for idx, (diam, slice) in enumerate(diam_slices):
            if diam is not None:
                if diam > max_diam:
                    max_diam = diam
                    max_plane = plane
                    max_index = idx
                    max_slice = slice
                else:
                    max_diam = max_diam
                    max_plane = max_plane
                    max_index = max_index
                    max_slice = max_slice
                  #  max_area = max_area
        
        area_str = None
        if max_diam in d1_values_and_cases:
            i = d1_values_and_cases.index(max_diam)
            area_str = str(area_values[i]) 
        elif max_diam in d2_values_and_cases:
            i = d2_values_and_cases.index(max_diam)
            area_str = str(area_values[i]) 

        if area_str is None:
            for idx, val in enumerate(d1_values_and_cases):
                if isinstance(val, str) and "and" in val:
                    if str(round(max_diam, 2)) in val.replace(" ", ""):
                        area_str = str(area_values[idx])
                        break
            
        if area_str:
            print(area_str)
            if "and" in area_str:
                try:
                    parts = [float(a.strip()) for a in area_str.split("and")]
                    lesions_2 = "yes"
                    max_area = round(sum(parts), 2)
                except ValueError:
                    max_area = None
            else:
                try:
                    max_area = float(area_str)
                    lesions_2 = "no"
                except ValueError:
                    max_area = None

      #  diam_data[plane] = [d for d, _ in diam_slices]
      #  print(diam_data[plane])
      #  area_data[plane] = [a for _, a in diam_slices]
      #  print(area_data[plane])
 #  area_list = area_data[max_plane]
  #  area_value = area_list[max_index] if max_index is not None and max_index < len(area_list) else np.nan


    return pd.Series({
        'max_diam': max_diam,
        'area_pipeline': max_area,
        'pipeline_plane': max_plane,
        'pipeline_slice_number': max_slice,
        'more_lesions': lesions_2
    })




def stack_scans(df):
    def fmt(values):
        return ",".join(f"{v:.2f}" if pd.notnull(v) else "" for v in values)

    grouped = df.groupby("pat_id").agg({
        "total_scandates": lambda x: "-".join(map(str, x)),
        "pipeline_max_diam": fmt,
        "area_pipeline": fmt,
        "pipeline_slice_number": lambda x: ",".join(map(str, x)),
        "pipeline_plane": lambda x: ",".join(map(str, x)),
        'more_lesions': lambda x: ",".join(map(str, x))
       # "Axial_Area_(cm2)": fmt,
       # "Axial_Diameters_(cm2)": fmt,
       # "Slice_number_Axial": lambda x: ",".join(map(str, x)),

      #  "Sagittal_Area_(cm2)": fmt,
      #  "Sagittal_Diameters_(cm2)": fmt,
      #  "Slice_number_Sagittal": lambda x: ",".join(map(str, x)),

      #  "Coronal_Area_(cm2)": fmt,
      #  "Coronal_Diameters_(cm2)": fmt,
      #  "Slice_number_Coronal": lambda x: ",".join(map(str, x))
    }).reset_index()
    
    return grouped

 
''' MAX AREA PER PLANE'''

def parse_area(area_str):
    if area_str is None or pd.isna(area_str):
        return None, "no"

    if isinstance(area_str, str) and "and" in area_str:
        try:
            parts = [float(a.strip()) for a in area_str.split("and")]
            return round(sum(parts), 2), "yes"
        except ValueError:
            return None, "no"
    else:
        try:
            return float(area_str), "no"
        except ValueError:
            return None, "no"


def compute_largest_area_per_plane(row):
    plane_names = ['axial', 'sagittal', 'coronal']
    results = {}

    for plane in plane_names:
        area_col  = f'{plane.capitalize()}_Area_(cm2)'
        slice_col = f'Slice_number_{plane.capitalize()}'

        area_values  = float_value_no_and(row[area_col])
        slice_values = slice_value(row[slice_col])

        max_area = -1
        max_slice = None
        more_lesions = "no"

        for i, area_str in enumerate(area_values):
            area, lesions_flag = parse_area(area_str)

            if area is not None and area > max_area:
                max_area = area
                max_slice = slice_values[i] if i < len(slice_values) else None
                more_lesions = lesions_flag

        if max_area < 0:
            max_area = np.nan
            max_slice = np.nan

        results[f'{plane}_max_area'] = max_area
        results[f'{plane}_slice_number'] = max_slice
        results[f'{plane}_more_lesions'] = more_lesions

    return pd.Series(results)


def stack_scans_per_plane(df):

    def fmt(values):
        return ",".join(f"{v:.2f}" if pd.notnull(v) else "" for v in values)

    grouped = df.groupby("pat_id").agg({
        "total_scandates": lambda x: "-".join(map(str, x)),

        "axial_max_area": fmt,
        "axial_slice_number": lambda x: ",".join(map(str, x)),
        "axial_more_lesions": lambda x: ",".join(map(str, x)),

        "sagittal_max_area": fmt,
        "sagittal_slice_number": lambda x: ",".join(map(str, x)),
        "sagittal_more_lesions": lambda x: ",".join(map(str, x)),

        "coronal_max_area": fmt,
        "coronal_slice_number": lambda x: ",".join(map(str, x)),
        "coronal_more_lesions": lambda x: ",".join(map(str, x)),
    }).reset_index()

    return grouped



"#################################################################################################################################################" 

"""             ####   FUNCTION TO CREATE THE CROSS SECTIONAL 2D DATASET GIVEN THE PLANE   ####     """

"#################################################################################################################################################" 

def tumor_measurements_2d(plane, folder_tumor, output_folder_2D, data_path):
    """ Give the plane and folder of tumor masks, it will calculate the 2D measurements based on RAPNO criteria"""
    available_scandates = {}
    data = pd.read_csv(data_path)
    total_scandates = sorted(data["total_scandates"].tolist())  # Ensure chronological order

    default_keys = {
    "Axial_area": [],
    "Axial_diameters": [],
    "Slice_number_Axial": [],
    "Sagittal_area": [],
    "Sagittal_diameters": [],
    "Slice_number_Sagittal": [],
    "Coronal_area": [],
    "Coronal_diameters": [],
    "Slice_number_Coronal": [],
    "Number_of_lesions": []
}

    results_dict = defaultdict(lambda: defaultdict(lambda: {k: [] for k in default_keys}))
   
    
    # Loop over all files in the tumor folder
    for file in os.listdir(folder_tumor):
      #  print(file)
        if file.endswith(".nii.gz"):
            filename = file[:-7]
           # id, scandate = filename.split('_')  # CHANGE THIS PART OF CODE IF YOU HAVE ADDED AN ADDITIONAL PART TO THE NAME
            id, scandate, _ = filename.split('_')
         #   print( id, scandate)
            
            image_path = os.path.join(folder_tumor, file)
            img_data = nib.load(image_path).get_fdata()
        
            new_mask = get_neighbors(img_data)  # Reduce the number of points of mask
            #print("binary mask", np.unique(new_mask))


            # Process the different planes
            if plane == "Axial":
                slice_mask, num_slice = get_valid_slice(new_mask, 2)
                lesions_results = process_contours(slice_mask, plane, file, scandate, id, results_dict)
                
                if lesions_results and len(lesions_results) > 0:
                    total_area = 0
                    results_dict[id][scandate]["Number_of_lesions"].append(len(lesions_results))
                    
                    for lesion in lesions_results:
                        rapno_measure, diam1, diam2 = lesion
            
            # Append area or "-" if None or invalid
                        if rapno_measure is not None and isinstance(rapno_measure, (int, float)):
                            total_area += rapno_measure
            
            # Append diameters if valid, else "-"
                        if diam1 is not None and diam2 is not None:
                            if isinstance(diam1, (int, float)) and isinstance(diam2, (int, float)):
                                results_dict[id][scandate]["Axial_diameters"].append(f"{round(diam1, 2)} x {round(diam2, 2)}")
                                results_dict[id][scandate]["Slice_number_Axial"].append(num_slice)
                            else:
                                results_dict[id][scandate]["Axial_diameters"].append("- x -")
                                results_dict[id][scandate]["Slice_number_Axial"].append("-")
                        else:
                            results_dict[id][scandate]["Axial_diameters"].append("- x -")
                            results_dict[id][scandate]["Slice_number_Axial"].append("-")

                            # Save total area sum separately if needed
                    results_dict[id][scandate]["Axial_area"] = [total_area if total_area else "-"]
                    
                else:
                    results_dict[id][scandate]["Axial_area"].append("-")
                    results_dict[id][scandate]["Axial_diameters"].append("- x -")
                    results_dict[id][scandate]["Slice_number_Axial"].append("-")
                    results_dict[id][scandate]["Axial_area"] = "-"

                
            elif plane == "Sagittal":
                slice_mask, num_slice = get_valid_slice(new_mask, 2)
                lesions_results = process_contours(slice_mask, plane, file, scandate, id, results_dict)
                
                if lesions_results and len(lesions_results) > 0:
                    total_area = 0
                    results_dict[id][scandate]["Number_of_lesions"].append(len(lesions_results))
                    
                    for lesion in lesions_results:
                        rapno_measure, diam1, diam2 = lesion
            
            # Append area or "-" if None or invalid
                        if rapno_measure is not None and isinstance(rapno_measure, (int, float)):
                            total_area += rapno_measure
            
            # Append diameters if valid, else "-"
                        if diam1 is not None and diam2 is not None:
                            if isinstance(diam1, (int, float)) and isinstance(diam2, (int, float)):
                                results_dict[id][scandate]["Sagittal_diameters"].append(f"{round(diam1, 2)} x {round(diam2, 2)}")
                                results_dict[id][scandate]["Slice_number_Sagittal"].append(num_slice)
                            else:
                                results_dict[id][scandate]["Sagittal_diameters"].append("- x -")
                                results_dict[id][scandate]["Slice_number_Sagittal"].append("-")
                        else:
                            results_dict[id][scandate]["Sagittal_diameters"].append("- x -")
                            results_dict[id][scandate]["Slice_number_Sagittal"].append("-")

                            # Save total area sum separately if needed
                    results_dict[id][scandate]["Sagittal_area"] = [total_area if total_area else "-"]
                    
                else:
                    results_dict[id][scandate]["Sagittal_area"].append("-")
                    results_dict[id][scandate]["Sagittal_diameters"].append("- x -")
                    results_dict[id][scandate]["Slice_number_Sagittal"].append("-")
                    results_dict[id][scandate]["Sagittal_area"] = "-"

            elif plane == "Coronal":
                slice_mask, num_slice = get_valid_slice(new_mask, 2)
                lesions_results = process_contours(slice_mask, plane, file, scandate, id, results_dict)
                
                if lesions_results and len(lesions_results) > 0:
                    total_area = 0
                    results_dict[id][scandate]["Number_of_lesions"].append(len(lesions_results))
                    
                    for lesion in lesions_results:
                        rapno_measure, diam1, diam2 = lesion
            
            # Append area or "-" if None or invalid
                        if rapno_measure is not None and isinstance(rapno_measure, (int, float)):
                            total_area += rapno_measure
            
            # Append diameters if valid, else "-"
                        if diam1 is not None and diam2 is not None:
                            if isinstance(diam1, (int, float)) and isinstance(diam2, (int, float)):
                                results_dict[id][scandate]["Coronal_diameters"].append(f"{round(diam1, 2)} x {round(diam2, 2)}")
                                results_dict[id][scandate]["Slice_number_Coronal"].append(num_slice)
                            else:
                                results_dict[id][scandate]["Coronal_diameters"].append("- x -")
                                results_dict[id][scandate]["Slice_number_Coronal"].append("-")
                        else:
                            results_dict[id][scandate]["Coronal_diameters"].append("- x -")
                            results_dict[id][scandate]["Slice_number_Coronal"].append("-")

                            # Save total area sum separately if needed
                    results_dict[id][scandate]["Coronal_area"] = [total_area if total_area else "-"]
                    
                else:
                    results_dict[id][scandate]["Coronal_area"].append("-")
                    results_dict[id][scandate]["Coronal_diameters"].append("- x -")
                    results_dict[id][scandate]["Slice_number_Coronal"].append("-")
                    results_dict[id][scandate]["Coronal_area"] = "-"

            elif plane == "all":
                slice_mask_axial, num_slice_axial = get_valid_slice(new_mask, 2)
                slice_mask_coronal, num_slice_coronal = get_valid_slice(new_mask, 1)
                slice_mask_sagittal, num_slice_sagittal = get_valid_slice(new_mask, 0)
                #print("Axial", slice_mask_axial, num_slice_axial, "Coronal", slice_mask_coronal, num_slice_coronal, "Sagittal", slice_mask_sagittal, num_slice_sagittal)
                planes = {
                    "Axial": (slice_mask_axial, num_slice_axial),
                    "Sagittal": (slice_mask_sagittal, num_slice_sagittal),
                    "Coronal": (slice_mask_coronal, num_slice_coronal)
                }
                for p, (mask, num_slice) in planes.items():
                    lesions_results = process_contours(mask, p, file, scandate, id, results_dict)
                #    print("result", result)
                    #if result is None or not None:
                    if lesions_results and len(lesions_results) > 0:
                        total_area = 0
                        results_dict[id][scandate]["Number_of_lesions"].append(len(lesions_results))
                    
                        for lesion in lesions_results:
                            rapno_measure, diam1, diam2 = lesion
            
            # Append area or "-" if None or invalid
                            if rapno_measure is not None and isinstance(rapno_measure, (int, float)):
                                total_area += rapno_measure

                            if p == "Axial":
                                results_dict[id][scandate]["Axial_area"] = [total_area if total_area else "-"]
                                if diam1 is not None and diam2 is not None:
                                    if isinstance(diam1, (int, float)) and isinstance(diam2, (int, float)):
                                        results_dict[id][scandate]["Axial_diameters"].append(f"{round(diam1, 2)} x {round(diam2, 2)}")
                                        results_dict[id][scandate]["Slice_number_Axial"].append(num_slice)
                                    else:
                                        results_dict[id][scandate]["Axial_diameters"].append("- x -")
                                        results_dict[id][scandate]["Slice_number_Axial"].append("-")
                                else:
                                    results_dict[id][scandate]["Axial_diameters"].append("- x -")
                                    results_dict[id][scandate]["Slice_number_Axial"].append("-")

                            elif p == "Sagittal":
                                results_dict[id][scandate]["Sagittal_area"] = [total_area if total_area else "-"]
                                if diam1 is not None and diam2 is not None:
                                    if isinstance(diam1, (int, float)) and isinstance(diam2, (int, float)):
                                        results_dict[id][scandate]["Sagittal_diameters"].append(f"{round(diam1, 2)} x {round(diam2, 2)}")
                                        results_dict[id][scandate]["Slice_number_Sagittal"].append(num_slice)
                                    else:
                                        results_dict[id][scandate]["Sagittal_diameters"].append("- x -")
                                        results_dict[id][scandate]["Slice_number_Sagittal"].append("-")
                                else:
                                    results_dict[id][scandate]["Sagittal_diameters"].append("- x -")
                                    results_dict[id][scandate]["Slice_number_Sagittal"].append("-")
                                
                            elif p == "Coronal":
                                results_dict[id][scandate]["Coronal_area"] = [total_area if total_area else "-"]
                                if diam1 is not None and diam2 is not None:
                                    if isinstance(diam1, (int, float)) and isinstance(diam2, (int, float)):
                                        results_dict[id][scandate]["Coronal_diameters"].append(f"{round(diam1, 2)} x {round(diam2, 2)}")
                                        results_dict[id][scandate]["Slice_number_Coronal"].append(num_slice)
                                    else:
                                        results_dict[id][scandate]["Coronal_diameters"].append("- x -")
                                        results_dict[id][scandate]["Slice_number_Coronal"].append("-")
                                else:
                                    results_dict[id][scandate]["Coronal_diameters"].append("- x -")
                                    results_dict[id][scandate]["Slice_number_Coronal"].append("-")

            else:
                raise ValueError("Invalid plane specified. Choose from 'Axial', 'Sagittal', 'Coronal', or 'all'.")

    # Prepare the results for saving into a CSV
    results = []
    print("results_dict done")
    for id, scan_data in results_dict.items():
        id = str(id).zfill(2)
    
    # Sort scandates chronologically (assumes sortable format like YYYYMMDD or YYYY-MM-DD)
        sorted_scandates = sorted(scan_data.keys())
    
        axial_areas = []
        axial_diameters = []
        axial_slices = []
    
        sagittal_areas  = []
        sagittal_diameters = []
        sagittal_slices = []
    
        coronal_areas = []
        coronal_diameters = []
        coronal_slices = []
        num_lesions_list = []

        for scandate in sorted_scandates:
            data = scan_data[scandate]
        
            axial_areas = data.get("Axial_area", []) if isinstance(data.get("Axial_area"), list) else [data.get("Axial_area")]
            axial_d1 = data.get("Axial_d1", [])
            axial_d2 = data.get("Axial_d2", [])
            axial_slices = data.get("Slice_number_Axial", [])

            sagittal_areas = data.get("Sagittal_area", []) if isinstance(data.get("Sagittal_area"), list) else [data.get("Sagittal_area")]
            sagittal_d1 = data.get("Sagittal_d1", [])
            sagittal_d2 = data.get("Sagittal_d2", [])
            sagittal_slices = data.get("Slice_number_Sagittal", [])

            coronal_areas = data.get("Coronal_area", []) if isinstance(data.get("Coronal_area"), list) else [data.get("Coronal_area")]
            coronal_d1 = data.get("Coronal_d1", [])
            coronal_d2 = data.get("Coronal_d2", [])
            coronal_slices = data.get("Slice_number_Coronal", [])

            results.append([
                id, 
                scandate,
                ",".join(map(str, axial_areas)),
                ",".join(map(str, axial_d1)),
                ",".join(map(str, axial_d2)),
                ",".join(map(str, axial_slices)),
                ",".join(map(str, sagittal_areas )),
                ",".join(map(str, sagittal_d1)),
                ",".join(map(str, sagittal_d2)),
                ",".join(map(str, sagittal_slices)),
                ",".join(map(str, coronal_areas)),
                ",".join(map(str, coronal_d1)),
                ",".join(map(str, coronal_d2)),
                ",".join(map(str, coronal_slices)),
            ])

    print(results)

    columns = [
        "pat_id", "Number_of_lesions_per_plane",
        "Axial_Area_(cm2)", "Axial_Diameters_(cm2)", "Slice_number_Axial", 
        "Sagittal_Area_(cm2)", "Sagittal_Diameters_(cm2)", "Slice_number_Sagittal",
        "Coronal_Area_(cm2)", "Coronal_Diameters_(cm2)", "Slice_number_Coronal"
    ]
    
    area2D = pd.DataFrame(results, columns=columns)
    output_path = os.path.join(output_folder_2D, f"cross_sectional_2D_{plane}.csv")
    area2D.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    print("2D dataset created!!")
    return area2D


"#################################################################################################################################################" 

"""               ####    FUNCTION TO COMPUTE 3D VOLUME    ####      """

"#################################################################################################################################################" 


def compute_tumor_volume(mask_path):
    # Load the NIfTI segmentation mask
    mask = sitk.ReadImage(mask_path)
    
    # Convert to NumPy array
    mask_array = sitk.GetArrayFromImage(mask)
    #print("mask",mask_array)

    ## NO EDEMA -> specify the tumor region as 1 and everything else as 0 (including edema)
  #  mask_array[mask_array == 2] = 0  
    mask_array[mask_array >= 1] = 1
  #  mask_array[mask_array == 3] = 1


    binary_mask = (mask_array == 1).astype(np.uint8)

    # Count the number of pixels with value 1 (tumor region)
    tumor_voxel_count = np.sum(binary_mask)
    print("count voxels", tumor_voxel_count)

    # Get voxel spacing (in mm)
    spacing = mask.GetSpacing()  # (x, y, z) voxel size in mm
    #print(spacing)
    # Compute volume of a single voxel (voxel size in mm³)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    print("voxel volume", voxel_volume)

    # Compute total tumor volume (in cm³)
    tumor_volume = (tumor_voxel_count * voxel_volume )/1000

 #   print(f"Tumor voxel count: {tumor_voxel_count}")
    #print(f"Voxel size (mm³): {voxel_volume}")
 #   print(f"Tumor volume (cm³): {tumor_volume}")

    return round(tumor_volume, 2)


"#################################################################################################################################################" 

"""                         #### FUNCTION TO GENERATE 3D VOLUME DATASET ####            """

"#################################################################################################################################################" 

def tumor_3D_volume(input_folder, dataset_path, output_path_3D):
    """Processes tumor volumes and updates the dataset."""

    final_dataset = pd.read_csv(
        dataset_path,
        dtype={"RT_start_date": str, "RT_end_date": str, "pat_id": str}
    )

    temp_volume_dict = defaultdict(list)
    results = []

    for file in os.listdir(input_folder):
        if file.endswith("_mask.nii.gz"):
            filename = file[:-7]
            id, scandate_vol, _ = filename.split('_')
            id = float(id)
            full_path = os.path.join(input_folder, file)
            volume = compute_tumor_volume(full_path)
            print("volume", volume)
            temp_volume_dict[id].append((scandate_vol, volume))

    for id, scandate_volume_list in temp_volume_dict.items():
        sorted_volumes = [v for _, v in sorted(scandate_volume_list, key=lambda x: x[0])]
        formatted_volumes = ["-" if v == 0.0 else float(v) for v in sorted_volumes]
        results.append([id, formatted_volumes])   # unconditional — outside any matching_rows loop

    df_results = pd.DataFrame(results, columns=["pat_id", "3D_Volume_(cm3)"])

    # Normalize BOTH sides to the same representation before merging
    final_dataset["pat_id"] = final_dataset["pat_id"].astype(str).str.extract(r'(\d+)')[0].astype(int)
    df_results["pat_id"] = df_results["pat_id"].astype(str).str.extract(r'(\d+)')[0].astype(int)

    print("final_dataset pat_id:", final_dataset["pat_id"].tolist())
    print("df_results pat_id:", df_results["pat_id"].tolist())

    volume3D = final_dataset.merge(df_results, on="pat_id", how="inner")
    print("merged rows:", len(volume3D))

    volume3D = volume3D.drop_duplicates(subset=["pat_id"], keep="first")

    if 'Unnamed: 0' in volume3D.columns:
        volume3D = volume3D.drop(columns=['Unnamed: 0'])
    if "3D_Volume_(cm3)_x" in volume3D.columns:
        volume3D.rename(columns={"3D_Volume_(cm3)_x": "3D_Volume_(cm3)"}, inplace=True)
    if "3D_Volume_(cm3)_y" in volume3D.columns:
        volume3D.drop(columns=["3D_Volume_(cm3)_y"], inplace=True)

    output_path = os.path.join(output_path_3D, "volume3D.csv")
    volume3D.to_csv(output_path, index=False)

    return volume3D


## CASE OF NO DATAFRAME + EACH ROW FOR SCAN 

def tumor_3D_volume_no_dataset(input_folder, output_path_3D):
    """Computes 3D tumor volume from NIfTI files without needing a dataset CSV."""

    temp_volume_dict = defaultdict(list)

    for file in os.listdir(input_folder):
        if file.endswith("_mask.nii.gz"):
            filename = file[:-7]  # Remove ".nii.gz"
            try:
                id, scandate_vol, _ = filename.split('_')
            except ValueError:
                print(f"Skipping file with unexpected format: {file}")
                continue

            id = str(id).zfill(2)  # Pad patient ID to 2 digits if needed
            full_path = os.path.join(input_folder, file)
            volume = compute_tumor_volume(full_path)

            temp_volume_dict[id].append((scandate_vol, volume))

    # Build the output DataFrame
    results = []
    for id, scandate_volume_list in temp_volume_dict.items():
        for scandate, vol in sorted(scandate_volume_list, key=lambda x: x[0]):
          #  id2 = f'PNOC008-{id}'
            results.append({
                "pat_id": id,
                "scan_date": scandate,
                "3D_Volume_(cm3)": float(vol),

            })

    volume3D_df = pd.DataFrame(results)
    output_path = os.path.join(output_path_3D, "volume3D.csv")
    volume3D_df.to_csv(output_path, index=False)

    print(f"[INFO] Saved computed volumes to: {output_path}")
    return volume3D_df



"#################################################################################################################################################" 

"""                      RUN CODE ON OWN DATASETS AND FOLDERS            """

"#################################################################################################################################################" 

if __name__ == '__main__':
    import argparse

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description="RAPNO pipeline: 2D/3D tumor measurements from segmentation masks.")
    parser.add_argument("--mask_folder", required=True, help="Folder containing predicted tumor masks (*_mask.nii.gz)")
    parser.add_argument("--img_folder", required=True, help="Folder containing background MRI images for overlays")
    parser.add_argument("--dataset_path", required=True, help="Path to clinical demographics CSV")
    parser.add_argument("--output_folder", required=True, help="Folder where all pipeline outputs will be saved")
    parser.add_argument("--plane", default="all", choices=["all", "Axial", "Coronal", "Sagittal"],
                         help="Which plane(s) to compute 2D measurements for")
    parser.add_argument("--trial_name", default="trial", help="Label used in output filenames (e.g. clinical trial name)")
    args = parser.parse_args()

    folder_tumor = args.mask_folder
    img_folder = args.img_folder
    dataset_path = args.dataset_path
    plane = args.plane
    trial_name = args.trial_name

    output_folder_2D = args.output_folder
    output_path_3D = args.output_folder
    final_output_folder = args.output_folder
    overlap_folder = os.path.join(args.output_folder, "overlapped_img")
    os.makedirs(overlap_folder, exist_ok=True)

    # ----- Step 0: Normalize clinical dataset column names/date formats -----
    final_dataset_path = os.path.join(output_folder_2D, "Demographics_formatted.csv")
    dataset = normalize_columns(dataset_path, final_dataset_path)

    # ----- Step 1: Compute 2D cross-sectional measurements (all slices, per lesion) -----
    volume_all_slices = tumor_measurements_2d_all_slices(
        plane=plane, folder_tumor=folder_tumor, output_folder_2D=output_folder_2D, data_path=None
    )

    # Largest single diameter/area per scan (across all planes)
    area2D = volume_all_slices.copy()
    area2D[['pipeline_max_diam', 'area_pipeline', 'pipeline_plane',
            'pipeline_slice_number', 'more_lesions']] = volume_all_slices.apply(compute_largest_diam_and_area, axis=1)
    area2D_path = os.path.join(output_folder_2D, f"area2d_all_slices_{trial_name}.csv")
    area2D.to_csv(area2D_path, index=False)

    # Largest area per plane, one row per patient (all scans stacked). This is waht is used for the visual dashboard 
    area2D_per_plane = volume_all_slices.copy()
    area2D_per_plane[['axial_max_area', 'axial_slice_number', 'axial_more_lesions',
                       'sagittal_max_area', 'sagittal_slice_number', 'sagittal_more_lesions',
                       'coronal_max_area', 'coronal_slice_number', 'coronal_more_lesions']] = \
        area2D_per_plane.apply(compute_largest_area_per_plane, axis=1)
    area2D_per_plane = stack_scans_per_plane(area2D_per_plane)
    print(area2D_per_plane.head())
    area2D_per_plane_path = os.path.join(output_folder_2D, f"area2d_all_slices_per_plane_{trial_name}.csv")
    area2D_per_plane.to_csv(area2D_per_plane_path, index=False)

    # ----- Step 2: Compute 3D tumor volume -----
    volume3D = tumor_3D_volume(input_folder=folder_tumor, dataset_path=final_dataset_path, output_path_3D=output_path_3D)
    # volume3D = tumor_3D_volume_no_dataset(input_folder=folder_tumor, output_path_3D=output_path_3D)
    # ^ use this instead if you have no clinical dataset and want one row per scan rather than per patient
    print("Finished computing 3D volume")

    # ----- Step 3: Save mask/image overlays for visual QC -----
    pat_imgs = save_masks_img(mask_folder=folder_tumor, image_folder=img_folder)
    overlap_masks_img(
        pat=pat_imgs,
        mask_folder=folder_tumor,
        img_folder=img_folder,
        output_folder=overlap_folder,
        largest_slice_csv_path=area2D_path
    )


"#################################################################################################################################################" 

"""                        #### MERGE 3D AND 2D VOLUME DATASETS ####                """

"#################################################################################################################################################" 


def normalize_pat_id(s):
    digits = s.astype(str).str.strip().str.extract(r'(\d+)')[0]
    return digits.str.zfill(2)    # keeps "02" as a string

area2D_per_plane['pat_id'] = normalize_pat_id(area2D_per_plane['pat_id'])
volume3D['pat_id'] = normalize_pat_id(volume3D['pat_id'])



volumes = volume3D.merge(area2D_per_plane, on="pat_id", how="inner")
print(volumes.head(), area2D_per_plane)

if "total_scandates_x" in volumes.columns:
    volumes.rename(columns={"total_scandates_x": "total_scandates"}, inplace=True)

if "3D_Volume_(cm3)_x" in volumes.columns:
    volumes.rename(columns={"3D_Volume_(cm3)_x": "3D_Volume_(cm3)"}, inplace=True)

if "3D_Volume_(cm3)_y" in volumes.columns:
    volumes.drop(columns=["3D_Volume_(cm3)_y"], inplace=True)

volumes.to_csv("volume_and_cross-sectional.csv", index = False)


"#################################################################################################################################################" 

"""                      ####  CREATION OF ADDITIONAL VARIABLES  ####                """

"#################################################################################################################################################" 


""" 1. From RT end date to date of progression
-> negative values means that the patient received the scan before RT"""



# Create a list to store differences for each patient
all_distances = []

for _, row in volumes.iterrows():
    id = row['pat_id']
    end_RT = row['RT_end_date']  # Directly get the value
    scans = str(row['total_scandates']).split("-")  # Assuming "-" is the delimiter

    # Convert end_RT to datetime if it's valid
    if pd.notna(end_RT) and isinstance(end_RT, str):
        try:
            date2 = datetime.strptime(end_RT, "%Y%m%d") 
        except ValueError:
            print(f"Skipping invalid end_RT format: {end_RT}")
            date2 = None
    else:
        date2 = None  # Handle missing values

    patient_distances = []  # Store distances for each patient in a list

    for scan_date in scans:
        try:
            # Convert scan date to datetime
            date1 = datetime.strptime(scan_date, "%Y%m%d")

            # Only calculate difference if date2 is valid
            if date2:
                difference = (date1 - date2).days
            else:
                difference = np.nan  # Use NaN if no valid end_RT

            patient_distances.append(str(difference))  # Store as string for easy joining

        except ValueError:
            print(f"Skipping invalid scan_date format: {scan_date}")
            patient_distances.append("NaN")  # Handle invalid scan dates

    # Append the list of differences as a single string (or keep as a list)
    all_distances.append(",".join(patient_distances))  # Store as "diff1-diff2-diff3..."

# Assign the list to a new column in the DataFrame
volumes['From_RT_end_to_scan(days)'] = all_distances

print(volumes.head())



""" 2. RAPNO progression variable  """

col = ["axial_max_area", "sagittal_max_area", "coronal_max_area"]
vol_2Ds = {}


for id, row in volumes.iterrows():
    # Create a dictionary per ID for 'all' planes

    if pd.notna(row['RT_end_date']):
    
        pat_id = row['pat_id']
        scans = str(row['total_scandates']).split("-")
        scan_dates = [datetime.strptime(date, "%Y%m%d") for date in scans]
        total_months = (max(scan_dates) - min(scan_dates)).days / 30.44

        days_after_RT = row['From_RT_end_to_scan(days)']
        days_prog = [float(x) for x in days_after_RT.split(",")]
        #vol_list = [float(x) for x in ast.literal_eval(row['3D_Volume_(cm3)'])]  USE THIS IF IT'S NOT A LIST
        vol_list = [float(x) if x != "-" else float('nan') for x in row['3D_Volume_(cm3)']]
       # print(len(days_prog), len(vol_list))

        days_prog_filtered = [day for day, volume in zip(days_prog, vol_list) if volume != "-"]
        #print(days_prog_filtered)
      

        days_prog_filtered = [day for day, volume in zip(days_prog, vol_list) if volume != "-"]

        volumes.at[id, 'From_RT_end_to_scan(days)_available'] = ", ".join(map(str, days_prog_filtered))



        if plane == "all":
                vol_2Ds = {c: row[c] for c in col if pd.notna(row[c])}  # Dictionary per ID
            #    print(vol_2Ds)
        else:
                c = f"{plane}_Area_(cm2)"
                vol_2Ds = {c: row[c]}
                #print(vol_2Ds)  #  {'Axial_Area_(cm2)': '23.06'}

    # If working with all planes, split the volumes into lists and handle the baseline
        if plane == "all":
            baselines = {}
            for plane_name, area_str in vol_2Ds.items():
            # Convert the area string into a list of floats
                areas_list = [float(x) for x in area_str.split(",") if x.strip() not in ["", "-"]]
             #   print("areas list", len(areas_list))

                if len(areas_list) < 2:  ## minimum of 2 scans
                    continue

                baseline_index = 0
                for i in reversed(range(len(days_prog_filtered))):  ## first scan immediately before end for RT. (most recent scan)
                    if days_prog_filtered[i] < 0:
                        baseline_index = i
                        break
                
                if baseline_index < len(areas_list):  ## because the first value can be -

                    baselines[plane_name] = float(areas_list[baseline_index])
                else: 
                    baselines[plane_name] = float(areas_list[1])

                ## TAKE MRI AFTER RT and NO baseline 
                areas_list_filtered = [
                    vol for vol, day in zip(areas_list[1:], days_prog_filtered[1:]) if day > 0
                ]

                #print("filtered", areas_list_filtered)

                if len(areas_list_filtered ) < 1: ## because if i have 1 + the baseline will be fine 
                    continue


            # Extract the first area value (baseline) for each plane
                  # First value is the baseline
           # print(f"Baseline for {plane_name}: {baselines[plane_name]}")\

                areas_list_filtered_small = []
                dif_ratios_from_smallest = []

                ## IF the baseline tumor is after RT i count it
                if days_prog_filtered[0] > 0:  ## incorporate to count the smallest tumor because this will be after RT
                    areas_list_filtered_small = [baselines[plane_name]] + areas_list_filtered 
                    
                    for i in range(1,  len(areas_list_filtered_small)):  # Start from the second 
                        current_area = areas_list_filtered_small[i]
                   #     print("current", current_area)
              # Find the smallest area before the current one
                        previous_areas = areas_list_filtered_small[:i]
                        if len(previous_areas) >= 2: ## correct because i have also baseline (so minimum of baseline + scan)
                            smallest_before_current = min(previous_areas)  # Only consider previous values
                            #print("small_at_time", smallest_before_current)
                            ratio = (current_area - smallest_before_current) / smallest_before_current

                            dif_ratios_from_smallest.append(ratio)

                        else:
                            ratio = 0
                            dif_ratios_from_smallest.append(ratio)

                else:
                    areas_list_filtered_small = areas_list_filtered.copy()  ## LIST OF AREAS, case where i have first scan before RT end
                    #print(areas_list_filtered_small)

                    for i in range( 0,len(areas_list_filtered_small)): 
                        current_area = areas_list_filtered_small[i]
                        print("current", current_area)
              # Find the smallest area before the current one
                        previous_areas = areas_list_filtered_small[:i]
                        if len(previous_areas) >= 2:
                            smallest_before_current = min(previous_areas)  # Only consider previous values
                            print("small_at_time", smallest_before_current)
                            ratio = (current_area - smallest_before_current) / smallest_before_current

                            dif_ratios_from_smallest.append(ratio)

                        else:
                            ratio = 0  ## not previous one available. 
                            dif_ratios_from_smallest.append(ratio)
                
                print("from smallest", dif_ratios_from_smallest)


        
            # Ensure other_vol values are converted to floats before performing subtraction
                dif_ratios_baseline = []
                for other_vol_val in areas_list_filtered:
                # Calculate the difference for each volume value
                    ratio = round((baselines[plane_name] - other_vol_val) / baselines[plane_name], 2)
                    dif_ratios_baseline.append(ratio)
                
                print("from baseline", dif_ratios_baseline, dif_ratios_from_smallest, "id", pat_id)

            # Determine progression state based on differences
                if all(dif_ratio >= 0.25 for dif_ratio in dif_ratios_baseline) and total_months >= 2:
                    volumes.at[id, f"Final_RAPNO_prog_{plane_name}"] = "Partial response"

                elif all(dif_ratio >= 0.25 for dif_ratio in dif_ratios_from_smallest if dif_ratio != 0):  
                    ## not count 0 because they are when the smallest mri is the first one or because there isn't
                    volumes.at[id, f"Final_RAPNO_prog_{plane_name}"] = "Progressive disease"

                elif all(area == 0 for area in areas_list_filtered) and total_months >= 2:
                    volumes.at[id, f"Final_RAPNO_prog_{plane_name}"] = "Complete response"

                else:
                    volumes.at[id, f"Final_RAPNO_prog_{plane_name}"] = "Stable disease"
                


                ## GIVE A CLIASSIFICATION FOR A SINGLE MRI ONLY IF THEY ARE AFTER RT    
                prog_single = []
                prog_single.append("Baseline")
                for i, mri_ratio in enumerate(dif_ratios_baseline):
              
                    if mri_ratio >= 0.25 and mri_ratio != 1: ## avoid complete response
                        prog_single.append("Partial response")

                    elif dif_ratios_from_smallest[i] >= 0.25:
                        prog_single.append("Progressive response")

                    elif dif_ratios_from_smallest[i] == 0 or dif_ratios_baseline[i]== 0:
                        prog_single.append("Complete response")
                    else:
                        prog_single.append("Stable disease")
                
                volumes.loc[id, f"RAPNO_prog_{plane_name}"] = ", ".join(prog_single)


        elif plane in ["Axial", "Coronal", "Sagittal"]:
        
       # print(f"Baseline tumor for {plane}: {baseline_tumor}")
            baseline_tumor = {}
            for plane_name, area in vol_2Ds.items():
                #print(vol_2Ds.items())
                #print(type(volume))
                area1 = area.split(",")
                if len(area1) < 2:  ## minimum of 3 volumes
                    continue
                #print("list", area)
                areas_list = [float(x) for x in area.split(",") if x.strip() != "-"]  # Ensure area is converted to list of floats


                areas_list_filtered = [
                        area for area, day in zip(areas_list[1:], days_prog_filtered[1:]) if day > 0
                    ]
                
                #print(areas_list_filtered)

                if len(areas_list_filtered) < 1:  ## minimum of 2 volumes
                    continue

                baseline_tumor[plane_name] = float(areas_list[0])
                #print(baseline_tumor[plane_name])

            # Calculate baseline differences from the first area (baseline_tumor) for the partial response
                dif_ratios_baseline = []
                for other_area_val in areas_list_filtered:
                # Calculate the difference for each area value
                    ratio = round((baseline_tumor[plane_name] - other_area_val) / baseline_tumor[plane_name], 2)
                    dif_ratios_baseline.append(ratio)
                print("from baseline", dif_ratios_baseline, dif_ratios_from_smallest, "id",id)

           
            ## Useful for progressive disease
                areas_list_filtered_small = []
                dif_ratios_from_smallest = []
                if days_prog_filtered[0] > 0:  ## incorporate to count the smallest tumor because this will be after RT
                    areas_list_filtered_small = [baseline_tumor[plane_name]] + areas_list_filtered 
                    
                    for i in range( 1, len(areas_list_filtered_small)):  # Start from the second element
                        current_area = areas_list_filtered_small[i]
                        print("current", current_area)
              # Find the smallest area before the current one
                        previous_areas = areas_list_filtered_small[:i]
                        if len(previous_areas) >= 2:
                            smallest_before_current = min(previous_areas)  # Only consider previous values
                            print("small_at_time", smallest_before_current)
                            ratio = (current_area - smallest_before_current) / smallest_before_current

                            dif_ratios_from_smallest.append(ratio)

                        else:
                            ratio = 0
                            dif_ratios_from_smallest.append(ratio)

                else:
                    areas_list_filtered_small = areas_list_filtered.copy()
                    print(areas_list_filtered_small)

                    for i in range( 0,len(areas_list_filtered_small)):  
                        current_area = areas_list_filtered_small[i]
                        print("current", current_area)
              # Find the smallest area before the current one
                        previous_areas = areas_list_filtered_small[:i]
                        if len(previous_areas) >= 2:
                            smallest_before_current = min(previous_areas)  # Only consider previous values
                            print("small_at_time", smallest_before_current)
                            ratio = (current_area - smallest_before_current) / smallest_before_current 

                            dif_ratios_from_smallest.append(ratio)

                        else:
                            ratio = 0
                            dif_ratios_from_smallest.append(ratio)
                
                print("from smallest", dif_ratios_from_smallest)

            # Determine progression state based on differences
                if dif_ratios_baseline and all(dif_ratio >= 0.25 for dif_ratio in dif_ratios_baseline) and total_months >= 2:  ## if it's an empty list, doesn't count
                    volumes.at[id, "Final_RAPNO_prog"] = "Partial response"

                elif dif_ratios_from_smallest and all(dif_ratio >= 0.25 for dif_ratio in dif_ratios_from_smallest if dif_ratio != 0):  
                    volumes.at[id, "Final_RAPNO_prog"] = "Progressive disease"

                elif areas_list[1:] and all(area == 0 for area in areas_list[1:]) and total_months >= 2:
                    volumes.at[id, "Final_RAPNO_prog"] = "Complete response"

                else:
                    volumes.at[id, "Final_RAPNO_prog"] = "Stable disease"

                
                prog_single = []
                prog_single.append("Baseline")
                for i, mri_ratio in enumerate(dif_ratios_baseline):
                    if mri_ratio <= -0.25 and mri_ratio != 1: ## avoid complete response
                        prog_single.append("Partial response")

                    elif dif_ratios_from_smallest[i] >= 0.25:
                        prog_single.append("Progressive response")

                    elif dif_ratios_from_smallest[i] == 0 or dif_ratios_baseline[i]== 0:
                        prog_single.append("Complete response")
                    else:
                        prog_single.append("Stable disease")
                
                volumes.loc[id, f"RAPNO_prog_{plane_name}"] = ", ".join(prog_single)

"""

for idx, row in volumes.iterrows():
    if pd.notna(row['RT_end_date']):
        pat_id = row['pat_id']

        # Parse scan dates
        scans = str(row['total_scandates']).split("-")
        scan_dates = [datetime.strptime(date, "%Y%m%d") for date in scans]
        total_months = (max(scan_dates) - min(scan_dates)).days / 30.44

        # Days after RT
        days_after_RT = row['From_RT_end_to_scan(days)']
        days_prog = [float(x) for x in days_after_RT.split(",")]

        # Convert area_pipeline string to list of floats
        #volumes_list = [
        #    float(x) if x.strip() not in ["", "-"] else float('nan')
        #    for x in str(row['area_pipeline']).split(",")
        #]

        # Convert area string to list of floats CASE OF AREA PER PLANE
        volumes_list_ax = [
            float(x) if x.strip() not in ["", "-"] else float('nan')
            for x in str(row['axial_max_area']).split(",")
        ]

        volumes_list_sag = [
            float(x) if x.strip() not in ["", "-"] else float('nan')
            for x in str(row['sagittal_max_area']).split(",")
        ]

        volumes_list_cor = [
            float(x) if x.strip() not in ["", "-"] else float('nan')
            for x in str(row['coronal_max_area']).split(",")
        ]

        volumes_list = [
            np.nanmax([ax, sag, cor])
            for ax, sag, cor in zip(
                volumes_list_ax,
                volumes_list_sag,
                volumes_list_cor
                )]


        # Filter out volumes before RT (day <= 0)
        days_prog_filtered = [
            day for day, vol in zip(days_prog, volumes_list) if day > 0 and not pd.isna(vol)
        ]
        volumes_list_filtered = [
            vol for vol, day in zip(volumes_list, days_prog) if day > 0 and not pd.isna(vol)
        ]

        # Skip if not enough scans
        if len(volumes_list_filtered) < 1:
            continue

        # Baseline is the first volume after RT
        baseline_volume = volumes_list_filtered[0]

        # Calculate difference ratios from baseline
        dif_ratios_baseline = [
            round((baseline_volume - v) / baseline_volume, 2)
            for v in volumes_list_filtered[1:]
        ]

        # Calculate difference ratios from smallest observed volume so far
        dif_ratios_from_smallest = []
        smallest_so_far = baseline_volume
        for v in volumes_list_filtered[1:]:
            ratio = (v - smallest_so_far) / smallest_so_far
            dif_ratios_from_smallest.append(ratio)
            if v < smallest_so_far:
                smallest_so_far = v

        # Assign Final RAPNO progression
        if dif_ratios_baseline and all(r >= 0.25 for r in dif_ratios_baseline) and total_months >= 2:
            volumes.at[idx, "Final_RAPNO_prog"] = "Partial response"
        elif dif_ratios_from_smallest and all(r >= 0.25 for r in dif_ratios_from_smallest if r != 0):
            volumes.at[idx, "Final_RAPNO_prog"] = "Progressive disease"
        elif all(v == 0 for v in volumes_list_filtered[1:]) and total_months >= 2:
            volumes.at[idx, "Final_RAPNO_prog"] = "Complete response"
        else:
            volumes.at[idx, "Final_RAPNO_prog"] = "Stable disease"

        # Assign single scan RAPNO progression
        prog_single = ["Baseline"]
        for i, r_base in enumerate(dif_ratios_baseline):
            r_small = dif_ratios_from_smallest[i]
            if r_base >= 0.25 and r_base != 1:
                prog_single.append("Partial response")
            elif r_small >= 0.25:
                prog_single.append("Progressive response")
            elif r_base == 0 or r_small == 0:
                prog_single.append("Complete response")
            else:
                prog_single.append("Stable disease")
        volumes.at[idx, "RAPNO_prog"] = ", ".join(prog_single)
               
"""

# Save the final DataFrame
volumes['pat_id'] = volumes['pat_id'].astype(str).str.extract(r'(\d+)')[0].str.zfill(2)  # Ensure pat_id is int for consistency
volumes.to_csv(os.path.join(final_output_folder, f"final_dataset_volumes_and_max_areas_{plane}.csv"), index=False)


