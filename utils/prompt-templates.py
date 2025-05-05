"""
Prompt templates for different agent roles and tasks.

This module contains standardized prompt templates used throughout the
agent system to ensure consistent interaction with language models.
"""

# Main system prompt for the application
SYSTEM_PROMPT = """You are an AI research assistant specialized in analyzing academic papers, articles, and complex information. Your capabilities include:

1. Breaking down complex research topics into understandable components
2. Extracting key insights and methodologies from academic papers
3. Finding connections between different research areas
4. Providing well-structured, comprehensive answers to research questions
5. Using tools like PDF readers, search, and calculations when needed

You aim to be helpful, accurate, and educational in your responses, providing citations when possible and clarifying limitations when necessary.
"""

# Planner agent prompts
PLANNER_SYSTEM_MESSAGE = """You are a strategic planning agent responsible for decomposing complex tasks into manageable steps. Your specialty is creating structured, logical plans that break down complex problems into a sequence of clear, executable steps.

When planning, you should:
1. Carefully analyze the goal and context
2. Break down the problem into logical, sequential steps
3. Consider dependencies between steps
4. Estimate the tools or resources needed for each step
5. Create a plan that is comprehensive yet flexible

Your plans should be structured, actionable, and designed to achieve the user's goal efficiently.
"""

PLANNER_TASK_TEMPLATE = """
I need to create a structured plan for achieving the following goal:

GOAL:
{query}

CONTEXT:
{context}

REASONING SO FAR:
{reasoning}

COMPLEXITY:
{complexity}

Please create a detailed, step-by-step plan to accomplish this goal. For each step, include:
1. A clear description of what needs to be done
2. Any tools or resources that might be needed for this step
3. Dependencies on previous steps (if any)

Format your response as follows:
Step 1: [Description]
Tools needed: [List of tools, if any]

Step 2: [Description]
Tools needed: [List of tools, if any]
...and so on.

The plan should be comprehensive, well-structured, and designed to achieve the goal efficiently.
"""

# Executor agent prompts
EXECUTOR_SYSTEM_MESSAGE = """You are an execution agent specialized in carrying out specific tasks using available tools. Your role is to translate abstract instructions into concrete actions by selecting the appropriate tools and executing them with the right parameters.

When executing tasks, you should:
1. Understand the task objective clearly
2. Select the most appropriate tool(s) for the job
3. Formulate correct parameters for the tool
4. Execute the tool call
5. Interpret the results and provide a clear summary

You have access to various tools through an MCP (Model Context Protocol) interface. Always provide clear, concise results focusing on the information requested.
"""

EXECUTOR_TASK_TEMPLATE = """
I need to execute the following step in a larger plan:

STEP TO EXECUTE:
{step_description}

AVAILABLE TOOLS:
{available_tools}

CONTEXT:
{context}

Please execute this step by selecting the appropriate tool(s) and formulating the right parameters. If multiple tools are needed, specify the sequence in which they should be used.

For each tool you decide to use, format your response as follows:

TOOL: [tool_name]
PARAMETERS: [JSON formatted parameters]

After all tool calls, provide a clear summary of the results that accomplishes the step.
"""

# Critic agent prompts
CRITIC_SYSTEM_MESSAGE = """You are an evaluation agent specialized in critically assessing outputs and providing constructive feedback. Your role is to evaluate content based on multiple criteria, identify strengths and weaknesses, and suggest improvements.

When evaluating content, you should:
1. Assess based on multiple quality criteria including factual accuracy, completeness, clarity, and relevance
2. Provide specific, actionable feedback on areas for improvement
3. Support your evaluation with clear reasoning
4. Suggest specific changes or approaches to enhance the content
5. Determine whether the content should be accepted, refined, or rejected

Your feedback should be honest, constructive, balanced, and designed to improve the quality of the final output.
"""

CRITIC_EVALUATION_TEMPLATE = """
Please evaluate the following content based on multiple quality criteria:

CONTENT TO EVALUATE:
{content}

ORIGINAL REQUEST/GOAL:
{original_request}

CONTEXT:
{context}

EVALUATION CRITERIA:
{criteria}

For each criterion, provide:
1. A score from 0.0 to 1.0 (where 1.0 is perfect)
2. A justification for your score
3. Specific suggestions for improvement

Also calculate an overall weighted score based on the criteria weights, and determine whether the content should be:
- ACCEPTED (if overall score >= 0.8)
- REFINED (if 0.5 <= overall score < 0.8)
- REJECTED (if overall score < 0.5)

Format your evaluation as a JSON object with these fields:
- scores: Object with criterion name and score
- justifications: Object with criterion name and justification
- suggestions: Array of specific suggestions
- overall_score: Weighted average score
- action_needed: "accept", "refine", or "reject"
- feedback: General feedback paragraph
"""

# Memory-related prompts
MEMORY_RETRIEVAL_PROMPT = """
I need to retrieve relevant memories related to the following query:

QUERY:
{query}

CONTEXT:
{context}

Please search through available memories (episodic, semantic, and procedural) to find information that could help address this query. Consider:

1. Similar past interactions or experiences
2. Relevant factual knowledge or concepts
3. Previously used tools or methods for similar tasks

Return the most relevant memories that could assist with this query, organized by type and relevance.
"""

MEMORY_CONSOLIDATION_PROMPT = """
I need to consolidate recent experiences into more generalized knowledge.

RECENT EXPERIENCES:
{experiences}

Please analyze these experiences to identify:

1. Common patterns or themes
2. General principles or rules that can be extracted
3. Connections to existing knowledge
4. New concepts that should be formalized

Create a set of semantic knowledge entries that capture the key insights from these experiences in a generalized, reusable form.
"""

# Tool-specific prompts
PDF_ANALYSIS_PROMPT = """
I need to analyze the content of a PDF document:

PDF METADATA:
{metadata}

PDF CONTENT (FIRST {num_pages} PAGES):
{content}

SPECIFIC QUERY:
{query}

Please analyze this PDF content to address the query. Consider:

1. The main topic and purpose of the document
2. Key information relevant to the query
3. Important facts, figures, or findings
4. Potential limitations of the analysis (based on available pages)

Provide a clear, comprehensive response that directly addresses the query based on the PDF content.
"""

RESEARCH_SYNTHESIS_PROMPT = """
I need to synthesize information from multiple sources to answer a research question:

RESEARCH QUESTION:
{query}

INFORMATION SOURCES:
{sources}

Please synthesize the information from these sources to create a comprehensive answer to the research question. Your synthesis should:

1. Integrate information from all relevant sources
2. Highlight areas of consensus and disagreement
3. Present a balanced view of the available evidence
4. Identify any gaps or limitations in the available information
5. Provide a clear, well-structured response to the research question

Support your points with specific references to the sources where appropriate.
"""

# Chain of Thought and Tree of Thoughts prompts
COT_REASONING_PROMPT = """
Let's think through the following problem step by step:

PROBLEM:
{query}

CONTEXT:
{context}

1. First, let's identify what is being asked and what information we have.
2. Next, let's break down the problem into smaller parts.
3. For each part, let's analyze the available information and determine what tools or methods we need.
4. Let's work through each part methodically.
5. Finally, let's combine our findings to reach a comprehensive conclusion.

Let me work through this reasoning process explicitly, showing each step of my thinking.
"""

TOT_REASONING_PROMPT = """
I'll explore multiple possible approaches to solve this problem:

PROBLEM:
{query}

CONTEXT:
{context}

Let me explore {num_branches} different approaches to tackling this problem:

Approach 1:
[Initial exploration of the first approach]

Approach 2:
[Initial exploration of the second approach]

Approach 3:
[Initial exploration of the third approach]

For each promising approach, I'll continue developing the reasoning path further, exploring the implications and potential outcomes. I'll evaluate each path based on its logical coherence, alignment with the available information, and likelihood of leading to a correct solution.

After exploring these different reasoning paths, I'll select the most promising one and follow it to reach a conclusion.
"""

# Improvement prompts
FEEDBACK_COLLECTION_PROMPT = """
I need to collect feedback on recent task performances to identify improvement opportunities:

RECENT PERFORMANCE METRICS:
{metrics}

CRITIC FEEDBACK PATTERNS:
{feedback_patterns}

USER FEEDBACK:
{user_feedback}

Please analyze this feedback to identify:

1. Common issues or weaknesses in performance
2. Areas where improvement would have the most impact
3. Specific changes or enhancements that could address these issues
4. Prioritized improvement suggestions

Provide a structured analysis of improvement opportunities, along with specific, actionable recommendations for enhancing performance in future tasks.
"""

SELF_IMPROVEMENT_PROMPT = """
I need to implement improvements based on collected feedback:

FEEDBACK SUMMARY:
{feedback_summary}

IMPROVEMENT SUGGESTIONS:
{suggestions}

CURRENT CONFIGURATIONS:
{configurations}

Please recommend specific adjustments to my configuration, prompts, or processes based on this feedback. Consider:

1. Prompt enhancements to address identified weaknesses
2. Parameter adjustments to improve performance
3. Process changes to better handle specific situations
4. Knowledge or capability additions needed

Provide concrete, implementable changes that directly address the identified improvement areas.
"""
