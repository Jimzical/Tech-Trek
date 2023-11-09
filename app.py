import streamlit as st
from helper_components import ColoredHeader, Notif

ColoredHeader(
    "Tech Trek | Fine Tuning Gpt’s",
    description=""
)

st.title("Introduction")
st.write('''You must have used the Chat With Document or Chat with Data
option in Chat-gpt. Discover how the dynamic combination of
GPT-3 and Llama-Index can create a new version of gpt models.
By the end of the workshop You’ll be able to create a new gpt
model whose responses would be specific to your dataset.
''')

st.title("Pre-requisites")
# modules required are openai streamlit, llama-index
st.write("1. OpenAI")
st.code("pip install openai")
st.write("2. Llama-index")
st.code("pip install llama-index")
st.write("3. Streamlit")
st.code("pip install streamlit")
st.write("4. Pypdf")
st.code("pip install pypdf")
st.write("5. OpenAI API key")
st.write("Check out this [Link](https://platform.openai.com/account/api-keys)")

ColoredHeader(
    "Lets Start Coding",
    description=" Lets start with the basic api usage"
)

st.code('''
import openai

# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY"

# Generate text
user_request = "Write a poem about a cat."
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=user_request
)

# Print the response
print(response.choices[0].text)

''')

st.title("Explanation")
st.divider()
st.subheader("1. Importing the openai module")
st.code(r"import openai")
st.markdown("*Needed to use the OpenAI API*")

st.subheader("2. Setting the OpenAI API key")
st.code("openai.api_key = \"YOUR_API_KEY\"")
st.markdown("*Replace YOUR_API_KEY with your API key*")

st.subheader("3. Choosing Prompt")
st.code('''user_request = "Write a poem about a cat."''')
st.markdown("*This is the prompt that will be used to generate the response*")
st.markdown("Check out")
st.markdown('''
* [ChatGPT Best Prompt Practices](https://platform.openai.com/docs/guides/gpt-best-practices/strategy-write-clear-instructions)
* [Prompt Examples](https://platform.openai.com/examples)
''')

st.subheader("4. Generating the response")
st.code(r'''
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=user_request
    )
''')
st.markdown("*The response is stored in the response variable, This is basically the response you would get on the ChatGPT website if you would type the same prompt in the chatbox*")
st.markdown('''
<h5>Explaining the Parameters</h5>

* `engine`, This is the engine being used, it depends on what you want to use the OpenAI for (text generation, audio model, image model, gpt3, gpt4, etc)
* `prompt`, This is the question or prompt you want to ask the model

''', unsafe_allow_html=True)

st.subheader("5. Printing the response")
st.code(r'''
print(response.choices[0].text)
''')
st.markdown("*This prints the first response generated to the console/terminal*")

ColoredHeader(
    "Lets Try Customizing It!",
    description="We will be using the llama-index module for this"
)

st.code('''
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

print("Setting up API Key...")
# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY"

def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert in analyzing student data of their particular University, Assume all input prompts to be with respect to the input data, Don't answer anything apart from educational related prompt"))
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

index = load_data()

# Initialize the chat engine
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
while True:
    prompt = input("Your question: ")
    if not prompt:
        break
    # Generate a response
    response = chat_engine.chat(prompt)
    print("Assistant:", response.response)
''')

st.title("Explanation")
st.divider()

st.subheader("1. Importing the Required Modules")

st.code("from llama_index import VectorStoreIndex, ServiceContext")
st.markdown("*Needed to use the llama-index module*")
st.markdown('''* `VectorStoreIndex`, in simple terms, is a tool that makes your text data understandable to a computer by converting it into numbers, and it does this with the help of a specialized language model (LLM). This can be useful for various applications like text analysis, searching, and machine learning.''')
st.markdown('''* `ServiceContext`, in simple terms, is like the control center for managing your text-related tasks. It's the part of the `llama_index` module that helps you set things up and keep track of what's going on.''')
st.code("from llama_index.llms import OpenAI")
st.markdown('''* `OpenAI`, in simple terms, is like a bridge to a very smart computer program. This computer program, called an OpenAI language model, is excellent at understanding and working with text.''')
st.code("from llama_index import SimpleDirectoryReader")
st.markdown('''* `SimpleDirectoryReader`, in simple terms, is a tool that helps you load text data from a directory on your computer.''')

st.subheader("2. Setting the OpenAI API key")
st.code("openai.api_key = \"YOUR_API_KEY\"")
st.markdown("*Replace YOUR_API_KEY with your API key*")

st.subheader("3. Loading the Data")
st.code(r'''
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert in analyzing student data of their particular University, Assume all input prompts to be with respect to the input data, Don't answer anything apart from educational related prompt"))
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index
    ''')
st.markdown("*This function loads the data from the data folder and returns the index*")


# MAY NEED TO BE SIMPLIFIED
st.divider()
st.markdown('''
 
This is the function `load_data()` where we are feeding our dataset folder into
llama_index and openai model, and generating vectors out of it.

1. `reader = SimpleDirectoryReader(input_dir="./data", recursive=True)`: 

This line creates an instance of the `SimpleDirectoryReader` class from Llama-Index. It specifies the
input directory as "./data" (Basically the folder where you custom datasets will be stored) and sets the `recursive` parameter to `True`, indicating that
it should recursively search for files in subdirectories as well.

2. `docs = reader.load_data()`: 

The `load_data()` function is called on the `reader`
object, which reads the data from the specified directory and returns a list of
`Document` objects. Each `Document` represents a file in the directory.

3. `service_context = ServiceContext.from_defaults()`:

This line creates a `ServiceContext` object using the `from_defaults()` method. It configures the context
with an OpenAI language model (`llm`) using the GPT-3.5 Turbo model. The `temperature`
parameter sets the randomness of the model's responses, and the `system_prompt`
provides a system-level prompt for the model.

4. `index = VectorStoreIndex.from_documents(docs, service_context=service_context)`:

The `from_documents()` method of `VectorStoreIndex` is called, passing in the `docs`
list and the `service_context`. This creates an instance of `VectorStoreIndex` and
populates it with the documents and their vector representations.

5. `return index`: 

The `index` object, which represents the populated
`VectorStoreIndex`, is returned from the `load_data()` function
''')

st.divider()
st.subheader("4. Initializing the Chat Engine")

st.code(r'''
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
''')

st.markdown("*This line initializes the chat engine*")
st.markdown('''
* `index` is the index we created in the previous step

* `as_chat_engine()` is a method of the `VectorStoreIndex` class that creates a chat engine from the index.
Basically all the Data which we had transformed into vectors in the previous step is now being used to create a chat engine.

* `chat_mode="condense_question"` This mode condenses the user's question(or prompt) into a shorter
form, which can be useful for more concise interactions. Basically removes the unnecessary words from the prompt

* `verbose=True` If this Parameter is set to True then the chat engine provides additional
information and feedback during the conversation process.
''')

st.divider()
st.subheader("5. Starting the Chat")
st.code(r'''
while True:
    prompt = input("Your question: ")
    if not prompt:
        break
    # Generate a response
    response = chat_engine.chat(prompt)
    print("Assistant:", response.response)
''')
st.markdown("*This Marks the Start of the Chatbot*")

st.markdown('''
* `while True:` 

This is a loop that runs forever until the user enters a blank prompt. This allows us to have a conversation with the chatbot continously.

* `prompt = input("Your question: ")` 

This line takes the user's input and stores it in the `prompt` variable. This is essentially the same as typing a prompt into the chat box on the ChatGPT website.

* `if not prompt:` 

This condition checks if the user has entered an empty prompt
(by pressing Enter without typing anything). If the prompt is empty, the loop is exited, and
the conversation ends

* `response = chat_engine.chat(prompt)` 

This line sends the user's prompt to the chat
engine for processing. The chat engine generates a response based on the input and assigns
it to the `response` variable.


* `print("Assistant:", response.response)` 

This line prints the assistant's response to the
console, indicating that it is the assistant's turn to speak. The `response.response`
attribute contains the actual response generated by the chat engine.
''')

ColoredHeader(
    "Streamlit",
    description="Now we will be using Streamlit to make a web app out of this. Let's start with the Basics"
)

st.subheader("1. Importing the streamlit module")
st.code("import streamlit as st")
st.subheader("2. How to Run Streamlit Code")
st.code('streamlit run app.py')
st.markdown("*This is the command to run a streamlit app, replace app.py with the your python file*")
st.caption("This command should be run instead of python app.py in the terminal, This will start a local server and open the app in your browser")


st.title("Text Elements")

st.subheader("1. Title")
with st.echo():
    st.title("This is how we can add Title") # similar to <h1> in html
st.divider()

st.subheader("2. Subheader")
with st.echo():
    st.subheader(':blue[Streamlit] is cool :rainbow[isnt it?]') # similar to <h2> in html ')

st.subheader("3. Write")
with st.echo():
    st.write("This is the *Easiest* way to items") 
    st.write("It is similar to how we use print() to show things in the terminal/commandline")
    st.write("It can be used to display any type of data")
st.divider()

st.subheader("4. Markdown")
with st.echo():
    st.markdown("<h3 style='color: red;'>Markdown Text</h3>", unsafe_allow_html=True)
    st.markdown('''
    Example of a table:
    | Name | Age |
    | --- | --- |
    | Alice | 20 |
    | Bob | 21 |
    | Carol | 22 |
    ''')
st.divider()

st.subheader("4. Write")
with st.echo():
    st.write("This is a _Write_, It displays any text element based on its data type") # to display any text element, will update based on the type of element
st.divider()

st.subheader("5. Latex")
with st.echo():
    st.latex(r"e^{i\pi} + 1 = 0") # to display any latex element
st.divider()

st.subheader("6. Caption")
with st.echo():
    st.caption('This is a string that explains something above.')
    st.caption('A caption with _italics_ :blue[colors] and emojis :sunglasses:')

st.subheader("7. Tables")
with st.echo():
    dct = {
        "Name": ["Alice", "Bob", "Carol"],
        "Age": [20, 21, 22]
    }
    st.table(dct)
    lst = [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ]
    st.dataframe(lst) # to display any text element, will update based on the type of element


st.title("Simple Graphs")

st.subheader("1. Line Chart")
with st.echo():
    st.line_chart([1, 2, 4, 5, 3]) # to display line chart

st.subheader("2. Area Chart")
with st.echo():
    st.area_chart([1, 2, 4, 5, 3]) # to display area chart

st.subheader("3. Bar Chart")
with st.echo():
    st.bar_chart([1, 2, 4, 5, 3]) # to display bar chart

st.caption("For More Graphs Check out the [Streamlit Docs For Graphs](https://docs.streamlit.io/library/api-reference/charts)")

st.title("Input Elements")

st.subheader("1. Text Input")
with st.echo():
    txt = st.text_input("This is a text input") # to get text input from the user
    st.write(txt)
    with st.expander("Documentation"):
        st.write(st.text_input)

st.subheader("2. Number Input")
with st.echo():
    num = st.number_input("This is a number input", value=10) # to get number input from the user
    st.write(num)
    with st.expander("Documentation"):
        st.write(st.number_input)

st.subheader("3. Button")
with st.echo():
    btn = st.button("This is a button, Click Me!") # to get button input from the user
    if btn:
        st.write("Button is clicked")
    with st.expander("Documentation"):
        st.write(st.button)

st.subheader("4. Checkbox")
with st.echo():
    check = st.checkbox("This is a checkbox") # to get checkbox input from the user

    if check:
        st.write("Checkbox is checked")
    else:
        st.write("Checkbox is unchecked")
    with st.expander("Documentation"):
        st.write(st.checkbox)

st.subheader("5. Toggle")
with st.echo():
    toggle = st.toggle("This is a toggle") # to get toggle input from the user
    if toggle:
        st.write("Toggle is on")
    else:
        st.write("Toggle is off")
    with st.expander("Documentation"):
        st.write(st.toggle)

st.subheader("6. Radio Button")
with st.echo():
    radio = st.radio("This is a radio button", options=["Option 1", "Option 2"]) # to get radio button input from the user
    st.write(radio)
    with st.expander("Documentation"):
        st.write(st.radio)

st.subheader("7. Selectbox")
with st.echo():
    select = st.selectbox("This is a selectbox", options=["James", "Bob", "Carol", "David", "Eleanor", "Frank", "Grace", "Helen", "Ian", "Julia","Alice","Jake","Charles"], index=None) # to get selectbox input from the user
    st.write(select)
    with st.expander("Documentation"):
        st.write(st.selectbox)
st.subheader("8. Multiselect")
with st.echo():
    multi = st.multiselect("This is a multiselect", options=["James", "Bob", "Carol", "David", "Eleanor", "Frank", "Grace", "Helen", "Ian", "Julia","Alice","Jake","Charles"]) # to get multiselect input from the user
    st.write(multi)
    with st.expander("Documentation"):
         st.write(st.multiselect)

st.subheader("9. Slider")
with st.echo():
    slider = st.slider("This is a slider", min_value=0, max_value=10) # to get slider input from the user
    st.write(slider)
    with st.expander("Documentation"):
        st.write(st.slider)

st.subheader("10. Date Input")
with st.echo():
    date = st.date_input("This is a date input") # to get date input from the user
    st.write(date)
    with st.expander("Documentation"):
        st.write(st.date_input)

st.divider()

st.title("Sidebar Elements")
st.markdown('''
    The elements in the sidebar are the same as the main page, 
    the only difference is that they are displayed in the sidebar
''')
side = st.toggle("Show Sidebar")

if side:
    with st.sidebar:
        ColoredHeader(
            "This is a Sidebar",
            description="This is a sidebar element. All the elements in the sidebar are the same as the main page"
            )
    st.sidebar.subheader("Usage")
    st.sidebar.code('''
        # method 1
        st.sidebar.title("This is a Sidebar")
        st.sidebar.subheader("This is a sidebar element")
        # method 2
        with st.sidebar:
            st.markdown("This is a sidebar markdown")
            st.caption("This is a sidebar caption")
    ''')
    st.sidebar.title("This is a Sidebar")
    st.sidebar.subheader("This is a sidebar element")
    st.sidebar.markdown("This is a sidebar markdown")
    st.sidebar.caption("This is a sidebar caption")

st.caption('''
For more info check out the [Streamlit Docs](https://docs.streamlit.io/en/stable/api.html)
''')
st.divider()

st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.markdown("<h2 style=\"color: gold; background: rgba(255, 215, 0, 0.1); border-radius: 10px; text-align: center;\">Thanks For Reading!</h2>", unsafe_allow_html=True)
# st.markdown("<h2 style=\"color: gold; background: rgba(255, 215, 0, 0.1); border-radius: 10px; text-align: center; text-decoration: underline;\">Thanks For Reading!</h2>", unsafe_allow_html=True)
