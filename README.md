# Reportr - An Agentic/Multi-Agent AI System  

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/7141cdb3-2f15-4bbe-9662-a18c925de0f5" />


## Motivation

The motivation of Reportr was to design and implement an Agentic AI system. The system is extensible. 
What I implemented is a tool that runs either in interactive mode (CLI-based) or in query mode.
The user specifies one or multiple research topics. The agent searches different sources like ArVix or Google Scholar for 
current, high-rated research papers, articles, blog-posts. It concurrrently downloads up to a maximum number
of files. Then it rates and summarizes them. The research results are eventually  shown on the screen and also stored 
in an .md document.


It is possible to configure Reportr to run periodically using time schedules like "all two week on Monday at 8:00". The documents are  partitioned in chunks and stored in a vector database. So far Reportr is not using this database, but
 can be easily extended with a query component that allows to ask questions about the stored content.


You may use Reportr with local or cloud-based LLMs. For local inference llama.cpp and Ollama are supported, but it is easy
to add other inference engines as well.
