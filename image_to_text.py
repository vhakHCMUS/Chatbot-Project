from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import matplotlib.pyplot as plt

model_name = "DGurgurov/im2latex"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = TrOCRProcessor.from_pretrained(model_name)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def clean_latex_formula(latex_formula):
    if r"\begin" in latex_formula and r"\end" in latex_formula:
        start_idx = latex_formula.find(r"}") + 1  # Tìm vị trí sau \begin{}
        end_idx = latex_formula.find(r"\end")    # Tìm vị trí \end
        latex_formula = latex_formula[start_idx:end_idx].strip()  # Lấy phần nội dung chính
    return latex_formula

def image_to_latex(image_path):
    # Xử lý ảnh
    image = preprocess_image(image_path)
    
    # Tạo input token từ ảnh
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    # Dự đoán với model
    generated_ids = model.generate(pixel_values)
    
    # Chuyển đổi ID sang mã LaTeX
    latex_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # simplify the latex formula for matplotlib rendering
    latex_output = clean_latex_formula(latex_output)
    return latex_output

image_path = "test.png"

latex_formula = image_to_latex(image_path)

print("Generated LaTeX Formula:")
print(latex_formula)

# Hiển thị công thức LaTeX bằng matplotlib
plt.figure(figsize=(6, 3))
plt.text(0.5, 0.5, f"${latex_formula}$", fontsize=20, ha='center', va='center')
plt.axis('off')
plt.show()
