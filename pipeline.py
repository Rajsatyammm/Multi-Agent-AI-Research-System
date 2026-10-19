from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

def run_research_pipeline(topic: str) -> dict:
    state = {}

    # search agent
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"find recent reliable and details information about the topic : {topic}")]
    })
    state['search_result'] = search_result['messages'][-1].content

    # reader agent
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [(
            "user", f"""Based on the following search results about {topic},
            pick the most relevent URL and scrape it for deeper content
            Search Results: {state['search_result'][:800]}"""
                      )]
    })
    state['scraped_content'] = reader_result['messages'][-1].content

    print(state['scraped_content'])

    research_combined = (
        f"SEARCH RESULTS: \n{state['search_result']} \n"
        f"DETAILED SCRAP CONTENT: \n{state['scraped_content']}"
    )

    # writer chain
    state['report'] = writer_chain.invoke({'topic': topic, "research": research_combined})

    # critic chain
    state['critic_report'] = critic_chain.invoke({"report": state['report']})

    print(state)
    return state

if __name__ == "__main__":
    topic = input("Enter topic name :: ")
    try:
        run_research_pipeline(topic)
    except Exception as e:
        print(f"Error occured :: {str(e)}")