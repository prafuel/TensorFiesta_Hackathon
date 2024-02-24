# this is independent folder
# this app will be running on other laptop, who is in same network as, cause in streamlit will be hosted on this laptop

from langchain.prompts import PromptTemplate
# from langchain_community.llms import huggingface_hub
from langchain.chains import LLMChain


from flask import Flask, request, render_template
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML",
#                     config={"max_new_tokens" : 256,
#                             "temperature":0.01})


def get_prompt(df):
    my_template = '''
        df={input}
        i want you to think like data analyst who is good at understanding line graphs which derived from following dataframe "dataframe = df", and your task is to 
        first analyze the dataframe it and in result give me exact 4 short key points in context of business pointof view based on analysis in the following python list format 
        key_points= [
            key point 1,key point 2,key point 3,key point 4
        ]
        important, dont print anything, just print result in given format
    '''
    
    prompt = PromptTemplate(
    input_variables=['input'],
    template=my_template
    )

    return prompt.format(input=df)


@app.route("/", methods=['get', 'post'])
def main():
    if request.method == "POST":
        user_input = request.json['user']
        # user_input = request.form['user']

        prompt = get_prompt(user_input)

        # output = llm.generate([prompt])

        return {
            "output" : prompt
        }

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=8000)