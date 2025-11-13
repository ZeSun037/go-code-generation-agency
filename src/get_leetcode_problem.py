import requests
import os
import re
import time
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_filtered_slugs():
    url = "https://leetcode.com/api/problems/all/"
    response = requests.get(url)
    data = response.json()

    problems = data["stat_status_pairs"]
    filtered = [
        {
            "slug": p["stat"]["question__title_slug"],
            "title": p["stat"]["question__title"],
            "difficulty": p["difficulty"]["level"],
        }
        for p in problems
        if p["difficulty"]["level"] >= 3  # only hard
    ]

    print(f"Number of Questions(Hard) : {len(filtered)} ")
    return filtered


def get_problem_content(slug):
    url = "https://leetcode.com/graphql"
    query = """
    query getQuestionDetail($titleSlug: String!) {
      question(titleSlug: $titleSlug) {
        title
        content
        difficulty
      }
    }
    """
    variables = {"titleSlug": slug}
    json_data = {"query": query, "variables": variables}
    headers = {"Content-Type": "application/json"}

    res = requests.post(url, json=json_data, headers=headers)
    return res.json()["data"]["question"]


# save .txt to ./leetcode/problems/
def save_problem_as_txt(problem):
    title = problem["title"]
    title = title.replace(" ", "_")

    difficulty = problem["difficulty"]
    html_content = problem["content"]

    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(project_root, "leetcode", "problems")
    os.makedirs(save_dir, exist_ok=True)
    safe_title = re.sub(r'[\\/*?:"<>|]', "_", f"{title}")
    path = os.path.join(save_dir, f"{safe_title}.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n")
        f.write(f"Difficulty: {difficulty}\n\n")
        f.write(text)


def main():
    problems = get_filtered_slugs()
    for p in tqdm(problems[-30:]):  # only get the last 100 problems to test
        try:
            problem = get_problem_content(p["slug"])
            save_problem_as_txt(problem)
            time.sleep(1)
        except Exception as e:
            print(f"Error on {p['slug']}: {e}")


if __name__ == "__main__":
    main()
