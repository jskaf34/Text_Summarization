def test_model(model, text, summary): 
    output, prompt = model(text)

    dash_line = '-'.join('' for _ in range(100))
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}')