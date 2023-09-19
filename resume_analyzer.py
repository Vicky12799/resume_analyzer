import streamlit as st

def main(): 
    # st.header("header")
    st.title("File Upload Tutorial")
    menu = ["Home", "Dataset", "DocumentFiles","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader ("Home")
    elif choice == "Dataset":
        st.subheader ("Dataset")
    elif choice == "DocumentFiles":
        st.subheader("DocumentFiles")
    else:
        st.subheader ("About")
if __name__ == '__main__': 
    main()