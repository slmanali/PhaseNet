Исправление рис. 9–11 для статьи ComplexPhaseNet

Причина: старая версия ghosting_blur_analysis_ru.py выполняет zoom.plot(...) всегда при --line.
Удаление PNG и повторный запуск не устраняют эту строку: она рисуется заново.

1. На ноутбуке откройте терминал в ~/Documents/GitHub/PhaseNet, активируйте venv_phasenet.
   Скопируйте исправленный скрипт, например:
   cp ~/Downloads/ghosting_blur_analysis_ru_fixed.py tools/ghosting_blur_analysis_ru.py

2. Перегенерируйте Hard — профиль сохранится, бирюзовой линии в сравнениях НЕ будет:
   python tools/ghosting_blur_analysis_ru.py \
     --mode hard --index 17 --crop 80 355 340 705 \
     --line 95 475 205 475 --phenomenon inspection \
     --roi-note "Moving pedestrian with black shirt and green-banded cap"

3. Перегенерируйте Extreme:
   python tools/ghosting_blur_analysis_ru.py \
     --mode extreme --index 16 --crop 119 164 263 308 --phenomenon inspection \
     --roi-note "Bright shop sign"

4. Обновите статью локально (выполняйте в корне PhaseNet; укажите путь к загруженному DOCX):
   python ~/Downloads/update_vfi_article_figures.py \
     --article ~/Downloads/ComplexPhaseNet_RU_Updated_Hard_Extreme_Figures_Final.docx \
     --output ~/Downloads/ComplexPhaseNet_RU_Figures_Corrected.docx

Updater verifies corresponding metadata files and refuses an image rendered with an overlay line.
Both figures 9 and 11 contain only: Эталон, ComplexPhaseNet, RIFE, FILM.
Figure 10 is a separate Russian-language brightness-profile plot of these same four curves.
The script does not edit raw predictions or their original metrics.
