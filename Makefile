#  Проверка проекта Deepfake Detection
#
#  Одна команда:
#      make verify
#
#  Что произойдёт:
#      1. Установятся зависимости
#      2. Данные разобьются на train/test
#      3. Модель оценится, появятся графики и отчёт

.PHONY: verify install preprocess evaluate clean help

help:
	@echo ""
	@echo "  make verify      — полная проверка проекта (рекомендуется)"
	@echo "  make install     — только установить зависимости"
	@echo "  make preprocess  — только разбить данные на train/test"
	@echo "  make evaluate    — только оценить модель"
	@echo "  make clean       — удалить обработанные данные и графики"
	@echo ""

install:
	@echo " Устанавливаем зависимости..."
	pip install -r requirements.txt
	@echo " Зависимости установлены"
	@echo ""

preprocess:
	@echo " Готовим данные (разбиваем на train/test)..."
	jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb --inplace
	@echo " Данные готовы"
	@echo ""

evaluate:
	@echo " Оцениваем модель (считаем метрики, строим графики)..."
	jupyter nbconvert --to notebook --execute notebooks/05_evaluation.ipynb --inplace
	@echo " Оценка завершена"
	@echo ""

verify: install preprocess evaluate
	@echo ""
	@echo "------------------------------------------------------"
	@echo "   ГОТОВО. Проверка прошла успешно."
	@echo "------------------------------------------------------"
	@echo ""
	@echo "  Открой эти файлы, чтобы увидеть результаты:"
	@echo ""
	@echo "    📄  reports/verify_report.md           — отчёт с метриками"
	@echo "    📊  reports/figures/model_comparison.csv — сравнение моделей"
	@echo "    🖼   reports/figures/                    — графики (CM, ROC)"
	@echo ""
	@echo "  Хочешь посмотреть подробнее — открой ноутбук:"
	@echo ""
	@echo "    jupyter notebook notebooks/05_evaluation.ipynb"
	@echo ""
	@echo "------------------------------------------------------"

clean:
	@echo "→ Чистим обработанные данные и графики..."
	rm -rf data/processed
	rm -f  reports/figures/verify_*
	rm -f  reports/verify_report.md
	@echo " Готово"