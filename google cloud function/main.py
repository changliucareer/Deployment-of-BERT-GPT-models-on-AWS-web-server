def textGenerator(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    from flask import Flask, jsonify, request, render_template
    import numpy as np
    from transformers import pipeline
    def get_gpt3_model():
        def GPT3Generator(original_text, max_len):

            GPT3= pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
            result = GPT3(original_text, max_length=max_len, do_sample=True, temperature=0.9)

            return result[0]['generated_text']
    
        return GPT3Generator


    GPT3Generator = get_gpt3_model()

    # request_json = request.get_json()
    # if request.args and 'message' in request.args:
    #     return request.args.get('message')
    # elif request_json and 'message' in request_json:
    #     return request_json['message']
    # else:
    #     return f'Hello World!'

    request_json = request.get_json()

    if "s" in request.args:
        try:
            gpt3_output = GPT3Generator(request.args['s'],20)
            return jsonify({'input': {'s':request.args['s'],'max_len':20}, 'gpt3-prediction': gpt3_output})
        except:
            if len(request.args['s'])>0:
                if gpt3_output:
                    return jsonify({'success': 'false', 'message': gpt3_output})
                else:
                    return jsonify({'success': 'false', 'message': 'gpt3 has no output'})
            return jsonify({'success': 'false', 'message': 'Input s was not passed correctly.'})
    elif request_json and 's' in request_json:
        try: 
            #gpt3_output = GPT3Generator(request_json['s'],20)
            #return jsonify({'input': {'s':request_json['s'],'max_len':20}, 'gpt3-prediction': gpt3_output})
            return request_json
        except:
            return jsonify({'success': 'false', 'message': 'Input json was not passed correctly.'})

    return jsonify({'success': 'false', 'message': 'No Input s was found.'})

    
