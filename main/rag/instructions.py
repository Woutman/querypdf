import textwrap

INSTRUCTIONS_TEXT_EXTRACTION = (
    'Extract all text from the attached PDF file. The PDF file is a page from a larger PDF document. '
    'Return a list of items and their types, which can be either "RunningHead", "Title", "Subheading", "NarrativeText", "List", "Table", "Infographic", "Graph", "Header", "Footer", or "OtherText".\n'
    '- Header: Text belonging to a page header. Found at the very top of the page. Not always present.\n'
    '- RunningHead: Title that appears at the top of each page, but below the Header, in a document, often used to identify the chapter or section. Not always present.\n'
    '- Title: Main title of the page. Found at the top of the page, below the RunningHead. Not always present, but there can be only one per page.\n'
    '- Subheading: Heading of a subsection of text. Found in the main body of the page.\n'
    '- NarrativeText: Text that makes up the main body of the document and contains most of the information.\n'
    '- List: A list of items, denoted by bulletpoints. Can have a category at the top. Include this category.\n'
    '- Table: A table containing data. Return as CSV. Remove grouping translators from numbers.\n'
    '- Infographic: A visual representation of a concept or data. Return the main idea that is being conveyed.\n'
    '- Graph: A graph plot of some data. Draw conclusions from the data. Be thorough and focus on interesting relationships instead of being descriptive.\n'
    '- Footer: Text belonging to a page footer.\n'
    '- OtherText: Text not belonging to any other category.\n'
    'Only use these types.\n'
    'Each item of NarrativeText should contain all text belonging to the previous Subheading.'
    'Use markdown to improve legibility of the text and use double newlines to separate paragraphs.\n'
    'Return the results in a JSON object and nothing else.'
)

INSTRUCTIONS_REPHRASING = textwrap.dedent("""\
    You are an LLM that's part of a RAG pipeline for a chatbot. You handle the query rephrasing part of the RAG pipeline.
    You will be given an OpenAI message history object as input. Your task is to rephrase the final user message so the retrieval and reranking steps will perform better on it. 
    Take into account that the retrieval and reranking steps will only see the final user message, so make sure that all context relating to it is contained within it.
    Return only the rephrased user message as output.\
""")

INSTRUCTIONS_SUMMARIZATION = textwrap.dedent("""\
    You are an LLM that's part of a RAG pipeline for a chatbot. You handle the summarization part of the RAG pipeline. 
    You will be given a query and list of documents as input. 
    Your task is to parse the documents for information that's relevant to the query and answer it. Only use information that can be found in the documents. If you can't answer the query based on this information, ask for clarification instead.
    Write the summary from the perspective of a {role}. Use a tone of voice and vocabulary typically associated with this role. Don't mention your role.
    Use markdown to present the information in an appealing way. Don't use HTML tags.\
""")
