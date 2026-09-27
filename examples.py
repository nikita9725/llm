"""Labeled demonstration inputs for the routing pipeline."""

from dataclasses import dataclass

from schemas import Category


@dataclass(frozen=True, slots=True)
class RoutingExample:
    """One input with the category expected from the classifier."""

    title: str
    text: str
    expected_category: Category


ROUTING_EXAMPLES = [
    RoutingExample(
        "Сброс пароля",
        "Не могу войти в личный кабинет после смены телефона. Подскажите, как "
        "сбросить пароль и восстановить доступ?",
        Category.SUPPORT,
    ),
    RoutingExample(
        "Ошибка приложения",
        "Приложение закрывается при загрузке PDF. Версия 4.2 на Android 15. "
        "Какие шаги помогут устранить ошибку?",
        Category.SUPPORT,
    ),
    RoutingExample(
        "Отзыв о поиске",
        "Новый поиск стал заметно быстрее. Было бы удобно также фильтровать "
        "результаты по дате.",
        Category.FEEDBACK,
    ),
    RoutingExample(
        "Предложение по интерфейсу",
        "Предлагаю добавить тёмную тему: вечером интерфейс слишком яркий. "
        "В остальном обновление мне нравится.",
        Category.FEEDBACK,
    ),
    RoutingExample(
        "Задержка заказа",
        "Заказ задерживается уже четыре дня, а поддержка дважды закрыла обращение "
        "без ответа. Требую объяснить задержку и назвать новый срок доставки.",
        Category.COMPLAINT,
    ),
    RoutingExample(
        "Двойное списание",
        "С карты дважды списали оплату за одну подписку. Это недопустимо, верните "
        "лишний платёж и сообщите, когда деньги поступят.",
        Category.COMPLAINT,
    ),
    RoutingExample(
        "Запрос демонстрации",
        "Мы выбираем CRM для отдела из 30 менеджеров. Можно посмотреть демонстрацию "
        "и обсудить подходящий тариф?",
        Category.SALES,
    ),
    RoutingExample(
        "Покупка лицензий",
        "Хотим приобрести десять лицензий для команды. Расскажите о вариантах "
        "подписки и о том, как начать оформление.",
        Category.SALES,
    ),
    RoutingExample(
        "Вопрос о Python",
        "Чем список в Python отличается от кортежа и когда лучше использовать "
        "каждую из этих структур?",
        Category.GENERAL_QUESTION,
    ),
    RoutingExample(
        "Рабочее планирование",
        "Как подготовить повестку еженедельной встречи, чтобы команда успевала "
        "обсудить решения за 45 минут?",
        Category.GENERAL_QUESTION,
    ),
]

# Input-only view retained for the Day 2 prompt comparison utility.
SAMPLE_INPUTS = [(example.title, example.text) for example in ROUTING_EXAMPLES]
