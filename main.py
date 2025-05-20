# main.py

from qa import ask_question

print("🤖 Ask anything about Islam or the Quran (type 'exit' to quit)\n")

while True:
    q = input("You: ")
    if q.lower() in ["exit", "quit"]:
        break
    answer = ask_question(q)
    print(f"\n🕌: {answer}\n")
