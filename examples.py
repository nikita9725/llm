"""Labeled demonstration inputs for the routing pipeline."""

from dataclasses import dataclass

from schemas import Category


@dataclass(frozen=True, slots=True)
class RoutingExample:
    """One input with the category expected from the classifier."""

    title: str
    filename: str
    text: str
    expected_category: Category


ROUTING_EXAMPLES = [
    RoutingExample(
        "Сброс пароля",
        "01_support_password_reset.txt",
        "Не могу войти в личный кабинет после смены телефона. Подскажите, как "
        "сбросить пароль и восстановить доступ?",
        Category.SUPPORT,
    ),
    RoutingExample(
        "Ошибка приложения",
        "02_support_pdf_crash.txt",
        "Приложение закрывается при загрузке PDF. Версия 4.2 на Android 15. "
        "Какие шаги помогут устранить ошибку?",
        Category.SUPPORT,
    ),
    RoutingExample(
        "Отзыв о поиске",
        "03_feedback_search.txt",
        "Новый поиск стал заметно быстрее. Было бы удобно также фильтровать "
        "результаты по дате.",
        Category.FEEDBACK,
    ),
    RoutingExample(
        "Предложение по интерфейсу",
        "04_feedback_dark_theme.txt",
        "Предлагаю добавить тёмную тему: вечером интерфейс слишком яркий. "
        "В остальном обновление мне нравится.",
        Category.FEEDBACK,
    ),
    RoutingExample(
        "Задержка заказа",
        "05_complaint_delayed_order.txt",
        "Заказ задерживается уже четыре дня, а поддержка дважды закрыла обращение "
        "без ответа. Требую объяснить задержку и назвать новый срок доставки.",
        Category.COMPLAINT,
    ),
    RoutingExample(
        "Двойное списание",
        "06_complaint_double_charge.txt",
        "С карты дважды списали оплату за одну подписку. Это недопустимо, верните "
        "лишний платёж и сообщите, когда деньги поступят.",
        Category.COMPLAINT,
    ),
    RoutingExample(
        "Запрос демонстрации",
        "07_sales_demo.txt",
        "Мы выбираем CRM для отдела из 30 менеджеров. Можно посмотреть демонстрацию "
        "и обсудить подходящий тариф?",
        Category.SALES,
    ),
    RoutingExample(
        "Покупка лицензий",
        "08_sales_licenses.txt",
        "Хотим приобрести десять лицензий для команды. Расскажите о вариантах "
        "подписки и о том, как начать оформление.",
        Category.SALES,
    ),
    RoutingExample(
        "Вопрос о Python",
        "09_general_python.txt",
        "Чем список в Python отличается от кортежа и когда лучше использовать "
        "каждую из этих структур?",
        Category.GENERAL_QUESTION,
    ),
    RoutingExample(
        "Рабочее планирование",
        "10_general_planning.txt",
        "Как подготовить повестку еженедельной встречи, чтобы команда успевала "
        "обсудить решения за 45 минут?",
        Category.GENERAL_QUESTION,
    ),
]

# Input-only view retained for the Day 2 prompt comparison utility.
SAMPLE_INPUTS = [(example.title, example.text) for example in ROUTING_EXAMPLES]
