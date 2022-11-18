from docs.autogeneration.formatter import (
    create_page_area,
    create_title,
    extract_docstring_info_from_evaluator,
)
from credoai.lens.pipeline_creator import build_list_of_evaluators

if __name__ == "__main__":
    all_ev = build_list_of_evaluators()
    for evaluator in all_ev:
        try:
            doc = extract_docstring_info_from_evaluator(evaluator)
            page_name = evaluator.name.lower()
            with open(f"./pages/evaluators/{page_name}.rst", "w") as text_file:
                text_file.write(doc)
        except:
            print(f"{evaluator.name} docstring not found")
