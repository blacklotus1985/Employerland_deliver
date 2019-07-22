from flask import Flask,request,make_response
import os,json
import os
from Webhook_emp_class import eventi
#import pandas as pd
# Flask is used to develop a REST Api.
app = Flask(__name__)

#geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
def webhook():
    print('start')
    # get request from DialogFlow
    req = request.get_json(silent=True, force=True)

    print("Request:")
    print(json.dumps(req, indent=4))
    
    #Class of autolettura
    event = eventi(req)
    res = event.contextChatbot()

    # Handle the request from DialogFlow
    #res = processRequest(req)
    print(res)

    # Response to the DialogFlow.
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

if __name__ == '__main__':

    # using 8080 port to connect
    port = int(os.getenv('PORT', 8888))
    print("Starting app on port %d" % port)
    app.run(debug=False, port=port, host='0.0.0.0')
