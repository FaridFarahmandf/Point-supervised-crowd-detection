# راهنمای استفاده از دیتاست UCF-QNRF

این راهنما نحوه استفاده از پروژه برای دیتاست UCF-QNRF را توضیح می‌دهد.

## ساختار دیتاست

دیتاست UCF باید در مسیر زیر قرار گیرد:
```
data/UCF/
  ├── Train/
  │   ├── img_0001.jpg
  │   ├── img_0001_ann.mat
  │   ├── img_0002.jpg
  │   ├── img_0002_ann.mat
  │   └── ...
  └── Test/
      ├── img_0001.jpg
      ├── img_0001_ann.mat
      └── ...
```

## مراحل استفاده

### 1. تولید Cache فایل‌ها (LUDA)

ابتدا باید cache فایل‌ها را برای train و test تولید کنید:

```bash
# برای train
python LUDA_generate_UCF.py
# سپس در فایل، split را به 'test' تغییر دهید و دوباره اجرا کنید
```

یا می‌توانید به صورت دستی در فایل `LUDA_generate_UCF.py` خط 13 را تغییر دهید:
- برای train: `split = 'train'`
- برای test: `split = 'test'`

### 2. آموزش مدل

```bash
python train_UCF.py
```

مدل‌های آموزش دیده در مسیر `output/valmodels/UCF/h/nooff/` ذخیره می‌شوند.

### 3. تست مدل

```bash
python test_UCF.py
```

نتایج تست در مسیر `output/valresults/UCF/h/nooff/` ذخیره می‌شوند.

## تفاوت‌های UCF با ShanghaiTech

1. **ساختار فایل‌های mat**: UCF از `annPoints` استفاده می‌کند به جای `image_info`
2. **نام فایل‌ها**: فایل‌های mat در UCF به صورت `img_XXXX_ann.mat` هستند
3. **مسیر فایل‌ها**: تصاویر و annotation ها در یک پوشه قرار دارند

## تنظیمات پیش‌فرض

- `C.offset = False`: استفاده از offset غیرفعال است
- `C.scale = 'h'`: فقط ارتفاع پیش‌بینی می‌شود
- `C.num_scale = 1`: یک scale map
- `C.down = 1`: بدون downsampling

می‌توانید این تنظیمات را در فایل `train_UCF.py` تغییر دهید.

