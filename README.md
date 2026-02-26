# Reportr - An Agentic/Multi-Agent AI System  

## Motivation

The motivation of Reportr was to design and implement an Agentic AI system. The system is extensible. 
What I implemented is a tool that runs either in interactive mode (CLI-based) or in query mode.
The user specifies one or multiple research topics. The agent searches different sources like ArVix or Google Scholar for 
current, high-rated research papers, articles, blog-posts. It concurrrently downloads up to a maximum number
of files. Then it rates and summarizes them. The research results are eventually  shown on the screen and also stored 
in an .md document.


It is possible to configure Reportr to run using time schedules like "all two week on Monday at 8:00". The documents
are  partitioned in chunks and stored in a vector database. So far Reportr is not using this database, but
 can be easily extended with a query component that allows to ask questions about the stored content.
