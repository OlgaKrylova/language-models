import streamlit as st

# st.markdown("# Main page 🎈")
# st.sidebar.markdown("# Main page 🎈")

# # Contents of ~/my_app/pages/page_2.py
# import streamlit as st

# st.markdown("# Page 2 ❄️")
# st.sidebar.markdown("# Page 2 ❄️")

# # Contents of ~/my_app/pages/page_3.py
# import streamlit as st

# st.markdown("# Page 3 🎉")
# st.sidebar.markdown("# Page 3 🎉")

# selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
# page_names_to_funcs[selected_page]()

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Добрый день! 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Это довольно бесполезная подборка разных моделей машинного обучения  
        👈 Если что-то хочется посмотреть - модели можно выбрать справа
        """)

if __name__ == "__main__":
    run()