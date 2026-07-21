
"""                         DASHBOARD of RAPNO project              """



"""
This script creates a dashboard to visualize the final results of tumor segmentation model. 

It shows for each scan, the 2D volume with all measurements and the pogression of prediciton for each scan. 

INPUT:
The input is the csv named 'final_dataset_volumes' obtained by RAPNO_script. In addition, a olfder with all segmented mask tumor of a single patient 

OUTPUT:


"""

import pandas as pd
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_uploader as du
import os
import dash_auth
import dash_enterprise_auth as auth
import base64
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import datetime
import ast
import glob


"1. User Credientials"

# Dash app

VALID_USERNAME_PASSWORD_PAIRS = [  ### specify your username and password here
    ['', '']
]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__)
auth = dash_auth.BasicAuth(
app,
VALID_USERNAME_PASSWORD_PAIRS
)



"2. Specify datasets"

input_csv = "final_dataset_volumes_and_max_areas_all.csv" # output csv file from Auto RAPNO pipeline

images_folder = "ai-rapno_segmented_masks" # folder with all segmentation masks 

images_overlap_folder = "overlapped_img" # folder with all overlapped images for each patient (Segmentation mask overlapped with the original MRI scan)
##Read the data and filter 

data = pd.read_csv(input_csv)


if 'Date_of_Birth' not in data.columns:
    data['Date_of_Birth'] = data['Age']   ## use if we have different name for this variable 




" 3. APP LAYOUT "

"""
It represents the app components that will be displayed in the web browser and here is provided as a list.
 Componets were added to the list: an html.Div. 
The Div has a few properties, such as children, which we use to add text content to the page.

html - provides classes for all of the HTML tags, and the keyword arguments describe the HTML attributes 
           like style, className, and id
dcc - generates higher-level components like controls and graphs

"""


## Define the layout to show our images

app.layout = html.Div(children=[
    html.H1(children='Longitudinal MRIs analysis'), ## title of app
    html.Div(children='''
        A web page to visualize the status of your patient 
              
    ''', style={'marginBottom': '20px'}), 
    html.Div([
        dcc.Input(id='patient-id-input', type='text', placeholder='Enter Patient ID'),
        html.Button('Filter', id='filter-button', n_clicks=0)
    ]),  
    html.H2(children = 'Patient info:'),
    dash_table.DataTable( 
        id = 'patient-info-table',
        data=[],
        columns=[
        {"name": "Tumor Location", "id": "tumor_location"},
        {"name": "Gender", "id": "Sex"},
        {"name": "Age", "id": "Date_of_Birth"},
        {"name": " Clinical Trial", "id": "Trial"}
        ],
        style_cell={
        'textAlign': 'center',
        'padding': '5px',
        'fontSize': '15px',
        'fontWeight': 'bold',
        'whiteSpace': 'normal',
        'height': 'auto'
        },
        style_data={
        'color': 'black',
        'backgroundColor': 'white'
        },
        style_header={
        'backgroundColor': 'rgba(0, 0, 255, 0.8)',
        'color': 'white',
        'fontWeight': 'bolder',
        'textTransform': 'uppercase',
        'textAlign': 'center'
        },
        style_table={
        'overflowY': 'auto',  # Enables vertical scrolling
        'border': '1px solid black',  # Adds border to the table
        'width': '50%',  # Adjust table width
        'margin': 'auto'  
        }),
    html.H3(children = 'Trial info:'),
    html.Div(id='trial-status', style={'marginBottom': '20px'}),


    html.H2("Tumor Progression Over Time"), 
    html.H3("Tumor area over the time:"),
    dcc.Dropdown(
        id='area-type-dropdown',
        options=[
            {'label': 'Axial Area', 'value': "axial_max_area"},
            {'label': 'Sagittal Area', 'value': 'sagittal_max_area'},
            {'label': 'Coronal Area', 'value': 'coronal_max_area'}
        ],
        style={'width': '50%'}
    ),
    dcc.Graph(id = "time-overview"),

    html.H3("Tumor volume over the time:"),
    dcc.Graph(id = "time-volume-overview"),

    html.H2("Segmented areas:"),
    dcc.Dropdown(
        id='plane-dropdown',
        options=[
            {'label': 'Axial', 'value': 'Axial'},
            {'label': 'Sagittal', 'value': 'Sagittal'},
            {'label': 'Coronal', 'value': 'Coronal'}
        ],
        style={'width': '50%'}
    ),
    html.Div(id="sequence-mri"),
    dcc.Dropdown(
        id='area-type-dropdown-segmented',
        options=[
            {'label': 'Axial Area', 'value': 'Final_RAPNO_prog_axial_max_area'},
            {'label': 'Sagittal Area', 'value': 'Final_RAPNO_prog_sagittal_max_area'},
            {'label': 'Coronal Area', 'value': 'Final_RAPNO_prog_coronal_max_area'}
        ], 
        value='Final_RAPNO_prog_axial_max_area',
        style={'width': '50%'}
    ),

     html.H2("New tissue grown"),
     dcc.Store(id='images-overlap-folder', data=images_overlap_folder),

     dcc.Dropdown(
        id='scan_date',
        placeholder='Select scan comparison',
        options=[],  
        style={'width': '50%', 'margin': '10px'}, 
        multi = False
    ),

    html.Div(id='image-overlap', style={'margin-top': '20px'})  
     

])



## ------ Time Graph Area --- ###

@app.callback(
    Output(component_id='time-overview', component_property= 'figure'),
    Input(component_id='filter-button', component_property='n_clicks'),
    State(component_id='patient-id-input',component_property= 'value'),  ## only after a click, the dashboard change 
    Input(component_id='area-type-dropdown', component_property='value')
    )

def create_fig_area(n_clicks, patient_id, selected_volume):   
    if n_clicks and patient_id and selected_volume:
        #data['pat_id'] = data['pat_id'].astype('Int64').astype(str)
        patient_id_float = int(patient_id)
        filtered_data = data[data['pat_id'] == patient_id_float].copy()
        #filtered_data = filtered_data[["total_scandates", "Axial_Volume_(cm2)", "Sagittal_Volume_(cm2)","Coronal_Volume_(cm2)", "RT_start_date", "RT_end_date", "Progression_date"]]


        #filtered_data = pd.DataFrame(filtered_data)

        filtered_data["RT_start_date"] = pd.to_datetime(filtered_data["RT_start_date"], format='%m/%d/%Y', errors='coerce') ## change format based on what I have!!!

        filtered_data["RT_end_date"] = pd.to_datetime(filtered_data["RT_end_date"], format='%m/%d/%Y',errors='coerce')
        rt_start = pd.to_datetime(filtered_data["RT_start_date"].iloc[0], format='%m/%d/%Y',errors="coerce")
        rt_end   = pd.to_datetime(filtered_data["RT_end_date"].iloc[0], format='%m/%d/%Y',errors="coerce")


        if "Progression_date" in filtered_data.columns:
            filtered_data["Progression_date"] = pd.to_datetime(filtered_data["Progression_date"], format='%Y-%m-%d', errors='coerce')

        filtered_data["total_scandates"] = filtered_data["total_scandates"].astype(str).apply( 
            lambda x: x.split("-") if pd.notna(x) else [])

        filtered_data[selected_volume ] = filtered_data[selected_volume].astype(str).apply(
            lambda x: x.split(",") if pd.notna(x) else [])
        
  


        filtered_data = filtered_data.explode(["total_scandates", selected_volume])

        scan_dates = filtered_data["total_scandates"]
        scan_dates = pd.to_datetime(scan_dates, format='%Y%m%d', errors='coerce')
        scan_dates = scan_dates.reset_index(drop=True)


        areas = filtered_data[selected_volume]
        areas = areas.replace("-", 0).astype(float)
        areas = pd.to_numeric(areas, errors='coerce')
        areas = areas.reset_index(drop=True)


        # Combine into a new DataFrame
        plot_data = pd.DataFrame({"Scan Date": scan_dates, "Area(cm2)": areas})

        fig = go.Figure()

# Create Plotly figure
        fig.add_trace(go.Scatter(
           x=plot_data["Scan Date"], 
           y=plot_data["Area(cm2)"],
          mode="lines+markers",  # Adds both lines and markers
           marker=dict(symbol="circle", size=8),  # You can change the symbol type
             name="Area Trend"
            ))
        fig.update_traces( showlegend=True)

# Find NaN points (RT_start_date, RT_end_date, Progression_date)
        zero_points = plot_data[plot_data["Area(cm2)"] == 0]

# Add 'X' marker at NaN points
        fig.add_trace(go.Scatter(
            x=zero_points["Scan Date"],
          y=zero_points["Area(cm2)"],
             mode="markers+text",
            marker=dict(symbol="x", size=10, color="red"),
            textposition="top center",
            name="NaN tumor area"
            ))
       
        fig.add_trace(go.Scatter(
           x=[rt_start, rt_start],
           y=[0, 60],  # Full height of graph
           mode="lines",
           line=dict(color="green", dash="dash"),
           name="RT Start"
           ))

        fig.add_trace(go.Scatter(
           x=[rt_end, rt_end],
           y=[0, 60],  # Full height of graph
           mode="lines",
           line=dict(color="green", dash="dash"),
           name="RT End"
          ))

        if "Progression_date" in filtered_data.columns:
            fig.add_trace(go.Scatter(
            x=[filtered_data["Progression_date"].iloc[0], filtered_data["Progression_date"].iloc[0]],
            y=[0, 60],  # Full height of graph
            mode="lines",
            line=dict(color="orange", dash="dash"),
            name="Progression"
            ))
       
        fig.update_layout(
            title="Area Progression Over Time",
            xaxis_title="Scan Date",
            yaxis_title="Area (cm²)",
            title_x=0.5  # Center the title
            )

        return fig
    return go.Figure()


## ---- Time graph Volume --------- ##

@app.callback(
    Output(component_id='time-volume-overview', component_property= 'figure'),
    Input(component_id='filter-button', component_property='n_clicks'),
    State(component_id='patient-id-input',component_property= 'value')
    )

def create_fig_volume(n_clicks, patient_id):   ### not working ####
    if n_clicks and patient_id:
        patient_id_float = float(patient_id)
        filtered_data = data[data['pat_id'] == patient_id_float].copy()

        filtered_data["RT_start_date"] = pd.to_datetime(filtered_data["RT_start_date"], format='%m/%d/%Y', errors='coerce')

        filtered_data["RT_end_date"] = pd.to_datetime(filtered_data["RT_end_date"],format='%m/%d/%Y', errors='coerce')
        rt_start = pd.to_datetime(filtered_data["RT_start_date"].iloc[0], format='%m/%d/%Y',errors="coerce")
        rt_end   = pd.to_datetime(filtered_data["RT_end_date"].iloc[0], format='%m/%d/%Y',errors="coerce")

        if "Progression_date" in filtered_data.columns:
            filtered_data["Progression_date"] = pd.to_datetime(filtered_data["Progression_date"], format='%Y-%m-%d', errors='coerce')

        filtered_data["total_scandates"] = filtered_data["total_scandates"].astype(str).apply( 
            lambda x: x.split("-") if pd.notna(x) else [])



        scan_dates = filtered_data["total_scandates"]
        scan_dates = scan_dates.apply(lambda dates: [pd.to_datetime(date, format='%Y%m%d', errors='coerce') for date in dates])
        scan_dates = scan_dates.apply(lambda dates: [d.date() for d in dates])
        scan_dates = scan_dates.reset_index(drop=True)
        scan_dates = scan_dates.iloc[0]

        volumes = filtered_data["3D_Volume_(cm3)"].tolist()[0]
        volumes_list = ast.literal_eval(volumes)  # safely parse the string to a Python list
        volumes_float = [float(v) if v != '-' else 0 for v in volumes_list]
        volumes = pd.to_numeric(volumes_float, errors='coerce')

        
       


        # Combine into a new DataFrame
        if  np.isnan(volumes).all():
            print("Error: volumes list is empty")
            plot_data = pd.DataFrame({"Scan Date": scan_dates, "Volume(cm3)": [None] * len(scan_dates)})
        else:
            plot_data = pd.DataFrame({"Scan Date": scan_dates, "Volume(cm3)": volumes})

        fig = go.Figure()

# Create Plotly figure
        fig.add_trace(go.Scatter(
           x=plot_data["Scan Date"], 
           y=plot_data["Volume(cm3)"],
          mode="lines+markers",  # Adds both lines and markers
           marker=dict(symbol="circle", size=8),  # You can change the symbol type
             name="Volume Trend"
            ))
        fig.update_traces( showlegend=True)

# Find NaN points (RT_start_date, RT_end_date, Progression_date)
        zero_points = plot_data[plot_data["Volume(cm3)"] == 0]

# Add 'X' marker at NaN points
        fig.add_trace(go.Scatter(
            x=zero_points["Scan Date"],
          y=zero_points["Volume(cm3)"],
             mode="markers+text",
            marker=dict(symbol="x", size=10, color="red"),
            textposition="top center",
            name="NaN tumor volume"
            ))
       
        fig.add_trace(go.Scatter(
           x=[rt_start, rt_start],
           y=[0, 120],  # Full height of graph
           mode="lines",
           line=dict(color="green", dash="dash"),
           name="RT Start"
           ))

        fig.add_trace(go.Scatter(
           x=[rt_end, rt_end],
           y=[0, 120],  # Full height of graph
           mode="lines",
           line=dict(color="green", dash="dash"),
           name="RT End"
          ))
        
        if "Progression_date" in filtered_data.columns:
            fig.add_trace(go.Scatter(
                x=[filtered_data["Progression_date"].iloc[0], filtered_data["Progression_date"].iloc[0]],
                y=[0, 120],  # Full height of graph
                mode="lines",
                line=dict(color="orange", dash="dash"),
                name="Progression"
               ))
       
        fig.update_layout(
            title="Volume Progression Over Time",
            xaxis_title="Scan Date",
            yaxis_title="Volume(cm3)",
            title_x=0.5  # Center the title
            )

        return fig
    return go.Figure()

 ## ----- Table ------- ###       

## Explain the input for the app and outputs (it also means the associaiton with these components)
@app.callback(
    Output(component_id='patient-info-table', component_property= 'data'),
    Output('trial-status', 'children'),
    Input(component_id='filter-button', component_property='n_clicks'),
    State(component_id='patient-id-input',component_property= 'value')  ## only after a click, the dashboard change 
    )

def filter_patient_data(n_clicks, patient_id):
    if n_clicks and patient_id:
        try:
            patient_id_float = float(patient_id)
            filtered_data = data[data['pat_id'] == patient_id_float].copy()
            
            if filtered_data.empty:
                return [], "No patient found with this ID."
            
            if 'Trial' not in filtered_data.columns:
                return [], html.Div("No trial data available for this patient.", style={'color': 'orange'})
            else:
                trial_name = filtered_data["Trial"].values[0]

            # Example check: trial is full if name is "PNOC008" CHANGE THIS PART BASED ON YOUR DATA/TRIAL CRITERIA
            if trial_name == "PNOC008":  # or use a more dynamic condition
                message = f"""Trial {trial_name} focuses on High grade glioma (excluding Diffuse Intrinsic Pontine Glioma) and it uses 
                whole exome sequencing to indentify the correct treatment for  patient. It's not an interventional trial. """
            
            elif trial_name == "PNOC007":
                message = f"""Trial {trial_name} focuses on Diffuse Intrinsic Pontine Glioma and Midline Glioma and it tests a new immunotherapy (vaccine)
                It's an interventional trial. """
            else:
                message = f"Trial {trial_name} details unavailable."

            table_data = filtered_data[["tumor_location", "Sex", "Date_of_Birth", "Trial"]].to_dict('records')
            return table_data, message

        except ValueError:
            return [], "Invalid patient ID. Please enter a valid number."
    return [], ""




## -----   Sequence of MRIs ------ ###

@app.callback(
    Output(component_id='sequence-mri', component_property='children'),  ## for images
    Input(component_id='filter-button', component_property='n_clicks'),
    State(component_id='patient-id-input', component_property='value'),
    Input(component_id='plane-dropdown', component_property='value'),
    Input(component_id='area-type-dropdown-segmented', component_property='value')
    )

def filter_patient_mri(n_clicks, patient_id, selected_plane, selected_rano): #3 out input and what it changes



    folder = images_folder
    if not n_clicks or not patient_id:
        return [] 
    
    #print(f"Selected RANO Column: {selected_rano}")
    
    data_local = pd.read_csv(input_csv)
    data_local.columns = data_local.columns.str.strip()
    data_local['pat_id'] = data_local['pat_id'].astype(str).str.strip().str.split('.').str[0].str.zfill(2)

    filter_mri = []
    patient_id_str = str(patient_id).strip().split('.')[0].zfill(2)
    filtered_data = data_local[data_local['pat_id'] == patient_id_str].copy()
  
    #filtered_data = data[data['pat_id'].astype(str) == str(patient_id)].copy()
    #filtered_data.columns = filtered_data.columns.str.strip()

    single_prog = {"Axial":"RAPNO_prog_axial_max_area", "Coronal":"RAPNO_prog_coronal_max_area", "Sagittal":"RAPNO_prog_sagittal_max_area"}

    if selected_plane in single_prog:
        # Get the correct column based on the selected plane
        progression_column = single_prog[selected_plane]
    else:
        progression_column = None
    
    if progression_column is not None and progression_column in filtered_data.columns:
        # Assuming the column contains progression statuses like "baseline, Partial response, Partial response"
        progress_statuses = filtered_data[progression_column].iloc[0]  # Get the progression string
        if isinstance(progress_statuses, str):
            progress_list = [status.strip() for status in progress_statuses.split(',')]  # Split into a list and strip extra spaces
        else:
    # If progress_statuses is not a string, convert it to a string
            progress_list = [str(progress_statuses).strip()]
    else:
        progress_list = []


    if selected_rano is not None and selected_rano in filtered_data.columns:
        column_values = filtered_data[selected_rano]
        column_values = column_values.iloc[0]
    
    else:
        print(f"Error: 'selected_rano' is either None or not a valid column in the DataFrame")
        column_values = []  

    # -------------------------------
    # Slice number mapping
    # -------------------------------
    slice_col_map = {
        "Axial": "axial_slice_number",
        "Sagittal": "sagittal_slice_number",
        "Coronal": "coronal_slice_number"
    }

    slice_numbers = []
    slice_col = slice_col_map.get(selected_plane)

    if slice_col and slice_col in filtered_data.columns:
        raw_slice = filtered_data[slice_col]
        raw_slice = raw_slice.iloc[0]
        if isinstance(raw_slice, str):
            slice_numbers = [int(float(s.strip())) for s in raw_slice.split(',')]
        elif isinstance(raw_slice, (list, tuple)):
            slice_numbers = list(map(int, raw_slice))

    if not slice_numbers:
        return [html.P("No slice information available for this plane.")]


    if selected_plane is None or selected_plane == '':
        return [html.P("Select a valid plane....", style={"textAlign": "center", "fontSize":"20px", 'fontWeight': 'bold'})]
    
    if selected_rano is None or selected_rano == '':
        return [html.P("Select a RAPNO measure...", style={"textAlign": "center", "fontSize":"20px", 'fontWeight': 'bold'})]
    
    scan_dates_col = 'total_scandates'  # or the actual column in your CSV
    scan_dates_str = filtered_data[scan_dates_col].iloc[0]  # get the string of all scan dates
    scan_dates_list = [s.strip() for s in scan_dates_str.split('-')]  # split on '-' if your CSV uses that
    scan_info = list(zip(scan_dates_list, slice_numbers, progress_list))


    
    image_data = []
   # i=0
    for image in os.listdir(folder):
        if image.endswith(".png") and selected_plane in image:
            filename = image[:-4]  # Separate filename and extension
 
            parts = filename.split('_')  # Use max split of 1 to avoid extra errors
            id = parts[0]
            scandate = parts[1]
            img_slice = parts[4]

            try:
                img_slice = int(img_slice)
            except ValueError:
                continue


            scandate = pd.to_datetime(scandate, format="%Y%m%d").strftime("%Y-%m-%d")

            match = [prog for s_date, s_num, prog in scan_info if int(s_num) == img_slice and pd.to_datetime(s_date, format="%Y%m%d").strftime("%Y-%m-%d") == scandate]
            if not match:
                continue  # no matching row in CSV

            

            


           # pos_scan = scan_dates.index[scan_dates.apply(lambda x: scandate in x)]
        
           # pos_scan =int(pos_scan[0]) ## take always the first value because there is always one match
        
           # slice_number_at_pos_scan = split_slice_number_list[pos_scan] 
#

        
            


            if id == patient_id:
                image_path = os.path.join(images_folder, image)

                with open(image_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode("utf-8")


                 # Construct Base64 Image Source
                img_src = f"data:image/png;base64,{img_data}"
                #progression_label = progress_list[i] if i <= len(progress_list) else "unknown"
               # progression_label = progress_list[min(i, len(progress_list) - 1)]
                progression_label = match[0]
                image_data.append({"img_src": img_src, "scandate": scandate, "image": image, "progression_label": progression_label})
                

           #     i += 1


    sorted_image_data = sorted(image_data, key=lambda x: datetime.datetime.strptime(x['scandate'], '%Y-%m-%d'))

    # Append sorted images to filter_mri
    for img_info in sorted_image_data:
        filter_mri.append(html.Div(children = [
            html.Img(src=img_info["img_src"], style={"width": "400px", "margin": "8px"}), 
            html.P(f"Scan Date: {img_info['scandate']}", style={"textAlign": "center", "fontSize":"15px"}),
            html.P(f"Progression Status: {img_info['progression_label']}", style={"textAlign": "center", "fontSize":"15px"}),
        ], style={"display": "flex", "flexDirection": "column", "overflowX": "auto", "marginRight": "20px"}))

    return [
        html.Div(children=filter_mri, style={"display": "flex", "flexWrap": "wrap", "justifyContent": "flex-start", "gap": "10px"}),
        html.Div(children=[
            html.P(f"RAPNO evaluation: {column_values}", style={"textAlign": "center", "fontSize": "30px", "fontWeight":"bold", "color":"blue"})
        ], style={"marginTop": "20px", "flexDirection": "column"})
    ] if filter_mri else [html.P("No images found for this patient.")]



### -----   Overlapping images ------ ###


@app.callback(
    Output('scan_date', 'options'),
    Input('filter-button', 'n_clicks'),
    State('patient-id-input', 'value'),
    State('images-overlap-folder', 'data')
)
def update_scan_date_options(n_clicks, patient_id, images_overlap_folder):

    if not n_clicks or not patient_id:
        return []

    #patient_id_str = str(patient_id).replace('.', '_')
    patient_id_str = str(patient_id).strip().zfill(2)  # convert float → int → string

    scan_dates = []
    for filename in os.listdir(images_overlap_folder):
        if filename.startswith(patient_id_str):
            parts = filename.replace('.png', '').split('_')

            if len(parts) > 2:
                scan_dates.append({'label': f'{parts[1]} vs {parts[2]}', 'value': f'{parts[1]}|{parts[2]}'})

    return scan_dates





@app.callback(
    Output('image-overlap', 'children'),
    [Input('scan_date', 'value')],
    State(component_id='patient-id-input', component_property='value'),
    State('images-overlap-folder', 'data') 
)

def get_overlap_images(scan_date, patient_id, images_overlap_folder):
 

    if not scan_date or not patient_id:
        return [html.P("Select a valid scan date and patient ID.")] 
    
    date1, date2 = scan_date.split("|")
    print(f"DEBUG: date1: {date1}, date2: {date2}")
    scan_key = f"{date1}_{date2}"
    patient_id_str = str(patient_id).replace('.', '_')
  
    image_pattern = os.path.join(images_overlap_folder, f"{patient_id_str}_{scan_key}*.png")
    image_files = glob.glob(image_pattern)

 
    if not image_files:
        return [html.P("No overlapped image found for the selected scan date.")]
    
    img_path = image_files[0]
    with open(img_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('ascii')

    return html.Img(
        src=f'data:image/png;base64,{encoded_image}',
        style={'maxWidth': '80%', 'marginBottom': '20px'}
    )


    
   
 

if __name__ == "__main__":
    app.run(debug=True)  ## refresh automatically after a change

    app.run(host='0.0.0.0', port=8051)
