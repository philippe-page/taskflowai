from taskflowai.task import Task
from typing import List, Dict, Union, Optional, Tuple
import json, re, base64, os

class ChatTasks:

    @staticmethod
    def answer_question(llm, user_query, conversation_history=None, context=None):
        history_context = f"Conversation history: {conversation_history}\n" if conversation_history else ""
        additional_context = f"Additional context: {context}\n" if context else ""
        task = Task(
            role="question-answering assistant",
            goal="Synthesize available information to answer user's question in an exhaustive and detailed manner",
            context=f"{history_context}{additional_context}\n--------\nUser Query: {user_query}",
            instruction="Based on the provided information, conversation history, additional context, and the user's original query, generate a comprehensive response that addresses the user's needs or question.",
            llm=llm,
        )
        answer = task.execute()
        return answer
    
    @staticmethod
    def translate_content(llm, contents: Union[str, List[str]], languages: Optional[Union[str, List[str]]] = None) -> Union[Dict[str, str], List[Dict[str, str]]]:
        """
        Translate content to specified languages.

        Args:
            llm: The language model to use for translation.
            contents (Union[str, List[str]]): Text item(s) to be translated.
            languages (Optional[Union[str, List[str]]]): Target language(s). Defaults to ["French", "Spanish"].

        Returns:
            Union[Dict[str, str], List[Dict[str, str]]]: A dictionary or list of dictionaries containing translations.
                                    Each dictionary has language names as keys and translated content as values,
                                    including the original English text with the key "English".
        """
        # Normalize inputs
        if isinstance(contents, str):
            contents = [contents]
        if languages is None:
            languages = ["French", "Spanish"]
        elif isinstance(languages, str):
            languages = [languages]
        
        translated_contents = []
        
        for content in contents:
            translations = {"English": content}
            
            for language in languages:
                task = Task(
                    role="professional translator",
                    attributes="You are fluent in multiple languages and skilled at accurate translations.",
                    goal=f"Translate the given content from English to {language}",
                    context=f"Content to translate:\n{content}",
                    instruction=f"Please translate the above content from English to {language}, maintaining the original meaning, tone, and style. Use appropriate idioms and expressions in {language}, and preserve any technical terms or proper nouns as needed. Provide only the translated text without any additional comments.",
                    llm=llm,
                )
                translations[language] = task.execute().strip()
            
            translated_contents.append(translations)
        
        # If input was a single string, return a single dictionary
        if len(translated_contents) == 1:
            return translated_contents[0]
        
        return translated_contents


class ImageTasks:

    @staticmethod
    def describe_image(llm, image_data: str) -> str:
        """
        Describe the contents of an image using Claude's image understanding capabilities.

        Args:
            image_data (str): The base64 encoded image data.

        Returns:
            str: A description of the image contents.
        """
        task = Task(
            role="image analysis expert",
            goal="Describe the contents of the provided image in detail",
            attributes="You have excellent visual perception and can describe images accurately and comprehensively.",
            instruction="Please take a close look at the image I've provided and describe what you see in detail. Start by identifying the main subject or focus of the image. Then, mention any people, objects, or animals that catch your eye. Don't forget to describe the setting or background to give a sense of context. As you analyze the image, talk about the colors, lighting, and overall mood it conveys. If you notice any text or symbols, please point those out as well. Finally, comment on the composition and any interesting artistic elements you observe. Now provide a vivid and comprehensive description that really captures the essence of the image.",
            llm=llm,
            image_data=image_data
        )

        # Execute the task with the image data
        description = task.execute()
        return description.strip()

class ResearchTasks:

    @staticmethod
    def synthesize_report(llm, topic, search_results):
        task = Task(
            role="Research Assistant",
            goal=f"Synthesize search results into a coherent report on {topic}",
            context=f"Topic: {topic}\nAggregated search results:\n{search_results}",
            instruction=f"""
    Review the aggregated search results and extract the most relevant information to thoroughly address the given topic. Synthesize the key points and findings into a well-organized, coherent report. The report should have an introduction, clear headings for each main section, and a conclusion summarizing the main takeaways. Aim for a report of approximately 800 words. Focus on addressing the foundational research question or topic of '{topic}'.
            """, 
            llm=llm,
        )
        report = task.execute()
        return report

    @staticmethod
    def report_news(llm, news_article):
        task = Task(
            role="an experienced journalist known for your engaging and informative news reporting",
            goal="Your task is to report the news. General Guidance: Maintain an active and engaging tone throughout the summary: Use strong, descriptive verbs to convey action and keep the reader engaged. Vary sentence structure to create a dynamic flow. Incorporate relevant quotes to add credibility and human interest: Select quotes that capture the essence of the story or provide unique insights. Attribute quotes properly to their sources. Integrate quotes seamlessly into the narrative flow of the summary. Structure the report logically and coherently: Begin with a compelling lead that hooks the reader and encapsulates the main point of the story. Follow the inverted pyramid structure, presenting the most important information first and gradually providing supporting details. Use paragraphs and subheadings to organize the content and enhance readability. Maintain objectivity and accuracy: Stick to the facts presented in the original article, avoiding speculation or personal opinions. Verify names, dates, and other key details to ensure accuracy. Be expansive and write as much detail as possible.",
            context=f"Here is the News Article to report:\n{news_article}",
            instruction="Respond with a direct report on the news, citing your sources. Do not report on 'the article' or mention 'the article', because you are producing a final output report for a professional journal.s",
            llm=llm,
        )
        report = task.execute()
        return report.strip()
    
    @staticmethod
    def report_weather(llm, weather_data):
        task = Task(
            role="experienced meteorologist and weather reporter",
            goal="Provide a clear, concise, and informative weather report based on the given data",
            attributes="You are skilled at interpreting weather data and communicating it effectively to a general audience.",
            context=f"Weather Data:\n{weather_data}",
            instruction="Analyze the provided weather data and create a comprehensive weather report. Your report should start with a brief overview of the current conditions. Provide details on temperature, humidity, wind speed and direction, and precipitation (if any). Mention any notable weather patterns or phenomena. Include a short-term forecast (next 24-48 hours) if the data allows. Highlight any weather warnings or advisories, if applicable. Use clear, concise language that is easily understood by a general audience. Maintain an engaging and informative tone throughout the report. Present your report in a well-structured format, using paragraphs to separate different aspects of the weather. Do not mention 'the data' or 'the report' in your response; simply present the information as a direct weather report. Be as detailed and comprehensive as the provided data allows.",
            llm=llm,
        )
        weather_report = task.execute()
        return weather_report.strip()

    @staticmethod
    def create_search_queries(llm, context, max_queries=None):
        task = Task(
            role="web researcher and data archivist",
            goal="Generate unique, specific, and granular queries for searching the web",
            attributes="You are precise, analytical, and skilled at breaking down complex topics into specific search queries.",
            context=context,
            instruction=f"""
The queries should aim to find relevant information related to the given context. These are *web search* queries, please write specific queries for *granular* information, not just general questions. Queries are tuned for search, and are focused, specific, and relevant. Aim for queries that are likely to return useful code snippets or documentation. Make the queries detailed and targeted, avoiding broad or generic terms.
Return a list of 2-4 queries, with one query per line. Do not include any additional text, numbering, or formatting.

For example:
example query 1
example query 2
example query 3
{f'Return a maximum of {max_queries} queries.' if max_queries else ''}
            """,
            llm=llm,
        )
        result = task.execute()
        queries = result.strip().split('\n')
        
        # Clean the queries
        cleaned_queries = []
        for query in queries:
            # Remove numbering, quotation marks, and extra whitespace
            cleaned = re.sub(r'^\d+\.\s*', '', query)  # Remove numbering at the start
            cleaned = cleaned.strip().strip('"')  # Remove leading/trailing whitespace and quotes
            cleaned_queries.append(cleaned)
        
        return cleaned_queries
    
    @staticmethod
    def summarize_website(llm, content):
        task = Task(
            role="website content summarizer",
            attributes="You are efficient at extracting key information and presenting it clearly and concisely.",
            goal="Summarize the given website content concisely and accurately",
            context=f"Website Content:\n{content}",
            instruction="Analyze the provided website content and create an extensive and detailed report that captures the main points, key information, and fine details such as quotes or snippets. Make sure to highlight the key points, features, or arguments presented, as well as important details and nuances. If you come across any important data, statistics, or examples, be sure to include those as well in codeblocks or quotes. Keep your report extensive and informative. Aim to write as much as you can. Provide this professional documentation and research report without comments before or after.",
            llm=llm,
        )
        summary = task.execute()
        return summary.strip()

    @staticmethod
    def select_urls(llm, combined_results, user_query, num_urls=None):
        task = Task(
            role="research assistant specialized in URL selection",
            goal="Select the most relevant URLs from search results based on the user's query",
            context=f"Search Results:\n{combined_results}\n\nUser Query: {user_query}",
            instruction=f"""
Analyze the search results and the user's query. Select the **most relevant** URLs that are likely to contain information to answer the user's question or address their needs. Consider the following criteria: Relevance to the user's query, Credibility and authority of the source, Recency of the information (if applicable), and Diversity of sources (to get a well-rounded perspective).
Return a simple list of the URLs you believe to be worth reading, one URL per line. Do not include any additional text or formatting, do not number the list, and Do NOT fabricate links, ONLY return real selected links given to you.

For example:
https://example1.com
https://example2.com
https://example3.com
...

Do not comment before or after the list.
{f'Return only a maximum of {num_urls} URLs.' if num_urls else ''}
            """,
            llm=llm,
        )
        result = task.execute()
        selected_urls = result.strip().split('\n')
        
        # Clean the URLs
        cleaned_urls = []
        for url in selected_urls:
            # Remove numbering at the start and any surrounding quotes
            cleaned = re.sub(r'^\d+\.?\s*', '', url)  # Remove numbering at the start
            cleaned = cleaned.strip().strip('"').strip("'")  # Remove quotes and extra whitespace
            if cleaned:  # Only add non-empty URLs
                cleaned_urls.append(cleaned)
        
        return cleaned_urls
    
    @staticmethod
    def determine_weather_search_params(llm, user_input):
        task = Task(
            role="Weather Search Parameter Analyzer",
            goal="Analyze user input and determine appropriate parameters for a weather search",
            context=f"User Input: {user_input}",
            instruction="""
Based on the user's input, determine the appropriate parameters for a weather search. Return a JSON object with the following structure:

{
    "location": "string",
    "days": null or integer between 1 and 10,
    "include_current": boolean,
    "include_astro": boolean,
    "include_hourly": boolean,
    "include_alerts": boolean
}

Guidelines:
1. 'location' is required and should be a string representing the place to search for.
2. 'days' should be null if not specified, or an integer between 1 and 10.
3. All boolean fields should be set based on the user's request or implied needs.
4. If a parameter is not mentioned or implied, set boolean fields to false and 'days' to null.

Respond only with the valid JSON object, no other text.
            """,
            llm=llm
        )
        
        response = task.execute()
        return json.loads(response)

    @staticmethod
    def determine_serper_search_params(llm, user_input):
        task = Task(
            role="Serper Search Parameter Analyzer",
            goal="Analyze user input and determine appropriate parameters for Serper API searches",
            context=f"User Input: {user_input}",
            instruction="""
    Based on the user's input, determine the appropriate parameters for Serper API searches. Return a JSON object with the following structure:

    {
        "query": "string" or ["string", "string", ...],
        "search_type": "search" | "news" | "images" | "shopping",
        "num_results": null or integer,
        "date_range": null or "h" | "d" | "w" | "m" | "y",
        "location": null or "string"
    }

    Guidelines:
    1. 'query' is either a single string or a list of strings, representing the search query or queries.
    2. 'search_type' is required and should be one of the allowed values.
    3. 'num_results' should be null if not specified, or a positive integer.
    4. 'date_range' should be null if not specified, or one of the allowed values.
    5. 'location' should be null if not specified, or a string representing the location.
    6. Determine the parameters based on the user's input and implied needs.
    7. If a parameter is not mentioned or implied, set it to null.
    8. The 'query' can be a single string or a list of 1-5 strings. Each query should be unique, independently valid, and directly relevant to the user's input. Avoid redundant or overly similar queries.

    Respond only with the valid JSON object, no other text.
            """,
            llm=llm
        )
        
        response = task.execute()
        return json.loads(response)


class DocumentationTasks:

    @staticmethod
    def analyze_project(llm, file_structure, developer_guidance):
        task = Task(
            role="senior software architect and technical documentation specialist",
            attributes="""You have extensive experience in analyzing complex codebases, identifying architectural patterns, and creating comprehensive documentation for developers. Your expertise spans multiple programming languages, frameworks, and software design paradigms.""",
            goal="Conduct a thorough analysis of the project structure and codebase to inform the creation of developer documentation",
            context=f"""
    Project structure:
    {file_structure}
    -----------
    Developer Guidance:
    {developer_guidance}
            """,
            instruction="""
Perform a comprehensive analysis of the provided project structure and codebase. Your analysis should provide a deep understanding of the project's architecture, components, and complexities to guide the documentation process. Begin with a high-level project overview, identifying its primary purpose and functionality. Determine the main programming languages and frameworks used, and assess the overall architectural approach. This overview should set the stage for a more detailed examination of the project's structure and components. Examine the directory structure, paying close attention to how files and folders are organized. Identify key directories and their purposes, such as those containing source code, tests, configuration files, and existing documentation. Evaluate how well the project adheres to standard layout conventions for its primary language or framework. Delve into the core components of the project, describing their main functions and responsibilities. Analyze how these components interact with each other, noting any significant dependencies or relationships. This analysis should reveal the overall design philosophy of the project and how different parts work together to achieve its goals. Assess the code organization, looking for patterns in how functionality is divided and encapsulated. Identify any design patterns or architectural styles employed throughout the codebase. Evaluate the use of object-oriented principles, functional programming techniques, or other paradigms that shape the code structure. Investigate the project's reliance on external dependencies. Describe the purpose and impact of major third-party libraries or frameworks used. Consider how these choices influence the project's functionality, maintainability, and potential future development. Examine the project's approach to configuration and environment management. Describe how the application handles different deployment scenarios, such as development, testing, and production environments. Identify key configuration files and explain their roles in customizing the application's behavior. Evaluate the project's commitment to quality assurance by examining its testing infrastructure. Describe the types of tests present (e.g., unit, integration, end-to-end) and how they are organized. Look for any continuous integration or deployment configurations that automate the testing and release processes. Assess the current state of documentation within the project. This includes README files, inline code comments, API documentation, and any other existing developer resources. Identify areas where documentation is comprehensive and highlight gaps that may need to be addressed in the new documentation. Throughout your analysis, be attentive to potential challenges or complexities that developers might face when working with this codebase. These could include intricate algorithms, complex state management, performance-critical sections, or areas with high technical debt. Highlighting these aspects will help focus the documentation on areas where developers need the most guidance. Finally, evaluate the project's adherence to coding standards and best practices. Identify any unique conventions or patterns used consistently throughout the codebase. This information will be crucial for helping new developers understand and maintain the project's code style and quality standards. Provide your analysis in a well-structured format, using markdown for enhanced readability. Use headings, subheadings, and code blocks where appropriate to organize the information effectively. Your goal is to create a comprehensive overview that will serve as a foundation for detailed developer documentation.
            """,
            llm=llm
        )
        return task.execute()

    @staticmethod
    def list_chapters(llm, doc_plan, developer_guidance, group_subchapters=True):
        instruction = f"""
Generate a comprehensive table of contents for the documentation based on the documentation plan.
Limit to 8-12 main chapters. For the time being, omit installation guides, contribution guidelines, references, and changelogs.

Format:
- Use numbered chapters for main sections (e.g., '1.', '2.').
- {'Use bullet points (-)' if group_subchapters else 'Use numbered sub-chapters (e.g., 1.1., 1.2.)'} for subtopics within each chapter.
- Maintain consistent indentation for subtopics.

Do not comment before or after the list; return only the list of chapters.
        """
        
        task = Task(
            role="chapter organizer",
            goal="identify and organize chapters",
            context=f"Documentation plan: {doc_plan}\n\nDeveloper Guidance:\n{developer_guidance}",
            instruction=instruction,
            llm=llm
        )
        result = task.execute()
        
        if group_subchapters:
            # Split by main chapters and process
            chapters = re.split(r'\n\d+\.', result)
            return [chapter.strip() for chapter in chapters if chapter.strip()]
        else:
            # Split by main chapters and sub-chapters
            chapters = re.split(r'\n\d+\.(?!\d)', result)
            processed_chapters = []
            for chapter in chapters:
                if chapter.strip():
                    # Further split sub-chapters if present
                    sub_chapters = re.split(r'\n(\d+\.\d+\.)', chapter)
                    processed_chapter = sub_chapters[0].strip()
                    for i in range(1, len(sub_chapters), 2):
                        if i+1 < len(sub_chapters):
                            processed_chapter += f"\n{sub_chapters[i]}{sub_chapters[i+1].strip()}"
                    processed_chapters.append(processed_chapter)
            return processed_chapters

    @staticmethod
    def write_chapter(llm, chapter_name, file_structure, developer_guidance, existing_docs):
        task = Task(
            role="technical writer",
            goal=f"write the {chapter_name} chapter",
            attributes="You follow the core principles of clarity, conciseness, and consistency in your writing: use simple words and clear language, aim for one idea per sentence, keep sentences short (15-20 words), and use consistent terminology throughout. You also avoid redundancy by not repeating information already covered in other chapters. You avoid excessive listing, preferring paragraphs to bullets. You format your pages with visual hierarchy and a clear structure, using markdown syntax for bolding, italics, code snippets, and headings. Ensure you're detailed *and* accessible language, making it simple for people.",
            context=f"""
    File structure:
    {file_structure}

    Developer guidance:
    {developer_guidance}

    Existing documentation:
    {existing_docs}
            """,
            instruction=f"""
Write the content for the '{chapter_name}' chapter of the developer documentation, explaining the relevant components, their usage, and any important details. Consider the existing documentation provided and avoid repeating information that has already been covered in other chapters. Focus on providing new, relevant information specific to this chapter. If you need to reference information from other chapters, do so briefly and provide a clear link or reference to where the reader can find more detailed information on that topic.
            """,
            llm=llm
        )
        return task.execute()


