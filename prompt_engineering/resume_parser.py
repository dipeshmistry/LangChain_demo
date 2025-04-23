from langchain_openai import ChatOpenAI
from output_parser import swabhav_technology_parser, SwabhavTech, resume_parser, ResumeInfo
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.globals import set_debug
load_dotenv()

set_debug(True)

print(os.getenv("OPENAI_API_KEY"))

# ChatOpenAI.openai_api_key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(temperature=0.7)

resume_txt = """
Here’s a clean and professional format for your resume content:

---

**Divya Waghmare**  
📞 9326788446 | 📧 [wdivya47@gmail.com](mailto:wdivya47@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/divya-waghmare/) | 💻 [GitHub](https://github.com/D25WR)

---

### **Summary**  
Skilled in Core Java, J2EE, Spring Boot, Spring MVC, Microservices, SQL (MySQL, Oracle SQL), and frontend technologies (HTML, CSS, JavaScript, jQuery, Bootstrap, Tailwind CSS, and React). Passionate about building scalable, high-performance applications, solving complex challenges with clean, maintainable code, and seeking to contribute my expertise to innovative and impactful projects.

---

### **Technical Skills**  
- **Programming Languages:** Java, SQL, JavaScript, C++  
- **Frontend Technologies:** HTML5, CSS3, JavaScript, jQuery, Bootstrap, Tailwind CSS, React  
- **Databases:** MySQL, Oracle SQL  
- **Frameworks:** Spring Boot, Spring MVC, Hibernate  
- **Tools:** Eclipse, Spring Tool Suite, VS Code, SQL Developer  

---

### **Projects**

**Finance Tracker SaaS**  
*Full-Stack Web Application | Feb 2025 – Present*  
**Tech Stack:** Java, Spring Boot, React, MySQL, Stripe API  
- Developed a cloud-based Finance Tracker SaaS for users to manage expenses, income, and budgeting.  
- Implemented secure authentication, real-time data visualization, and AI-driven expense categorization.  
- Integrated Stripe API for seamless subscription-based payments and premium features.  
- Ensured high scalability and security using Spring Boot, JWT authentication, and cloud storage.

**Phonebook Application**  
*Full-Stack Web Application | Aug 2024 – Dec 2024*  
**Tech Stack:** Java, Spring Boot, MySQL, HTML, Bootstrap  
- Built CRUD operations with a user-friendly interface.  
- Designed a responsive and interactive UI using HTML, Bootstrap, and JavaScript.  
- Enhanced search functionality and implemented role-based access control for better security.

---

### **Experience**

**Frontend Developer Intern**  
*Experts Coder | Mar 2024 – Jun 2024 | Mumbai, Maharashtra*  
- Developed responsive web applications and contact pages using Tailwind CSS, React, and JavaScript.  
- Collaborated in a team environment to design and implement interactive user interfaces.  
- Gained hands-on experience in client-based project development.

---

### **Education**

**SNDT Women’s University, Mumbai**  
*B.Tech in Computer Engineering | Sep 2020 – Sep 2024*  

---

### **Certifications**

- **Java Full Stack Development** – QuickXpert Infotech

---

Let me know if you'd like this in a downloadable format (PDF/Word), or if you want to tailor it for a specific job or domain!"""

resume_prompt = PromptTemplate(
    template="""
Context: 
Resume:
---
""" + resume_txt + """
---

Task:
Given the following resume, extract structured data in the specified format.

Instructions: 
{format_instructions}
""",
    input_variables=["query"],
    partial_variables={"format_instructions": resume_parser.get_format_instructions()}
)

print(resume_parser.get_format_instructions())
# prompt = PromptTemplate(template=template, input_variables=["task"],partial_variables={"format_instructions":swabhav_technology_parser.get_format_instructions()})
formatted_prompt = resume_prompt.format(resume_text=resume_txt)
print(formatted_prompt)
chain = resume_prompt | llm | JsonOutputParser

query = "Given the following resume, extract structured data in the specified format."
response = chain.invoke(input={"query": query})
print(response)
