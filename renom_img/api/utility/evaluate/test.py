from renom_img.api.utility.evaluate import EvaluatorDetection
pred = [[{'box': [10, 20, 60, 80], 'score': 0.8, 'class': 0 },{'box': [70, 90, 120, 110], 'score': 0.9, 'class': 1},{'box': [15, 25, 60, 80], 'score': 0.6, 'class': 0 }],[{'box': [20, 40, 70, 90], 'score': 0.8, 'class': 0}]]

gt = [[{'box': [15, 25, 55, 70], 'class': 0 },{'box': [80, 95, 125, 105], 'class': 1}],[{'box': [35, 45, 75, 95], 'class': 0}]]


eval = EvaluatorDetection(pred, gt)
print(eval.AP())
