from flask import Flask, request, render_template
import jinja2
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')


from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('C:/Users/User/Documents/logistics 2'))
template = env.get_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the form inputs
    country = int(request.form['Country'])
    vendor_inco_term = int(request.form['Vendor INCO Term'])
    shipment_mode = int(request.form['Shipment Mode'])
    line_item_quantity = int(request.form['Line Item Quantity'])

    pack_price = int(request.form['Pack Price'])
    unit_price = int(request.form['Unit Price'])
    Manufacturing_Site=int(request.form['Manufacturing Site'])
    weight_kilograms = int(request.form['Weight (Kilograms)'])
    fulfill_via_rdc = int(request.form['Fulfill Via_From RDC'])
    managed_by_pmo_us = int(request.form['Managed By_PMO - US'])
    product_group_hrdt = int(request.form['Product Group_HRDT'])
    product_group_uncommon = int(request.form['Product Group_uncommon'])
    First_Line_Designation_Yes=int(request.form['First Line Designation_Yes'])
    # Prepare the input array for prediction
    
    X = np.array([[country,vendor_inco_term,shipment_mode,line_item_quantity,pack_price,unit_price,Manufacturing_Site,weight_kilograms, 
                   fulfill_via_rdc,managed_by_pmo_us,product_group_hrdt,product_group_uncommon,First_Line_Designation_Yes]])
    
    
    # Make the prediction
    prediction = model.predict(X)[0]
    
    return render_template('home.html', prediction_text='Consisment Price should be $ {}'.format(prediction))


# @app.route('/predict',methods=['POST'])
# def predict():
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = reg_model.predict(final_features)

    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(prediction))


    # Return the result to the user
    # return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
