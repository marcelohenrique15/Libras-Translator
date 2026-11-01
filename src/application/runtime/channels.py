"""Troca limitada de mensagens: atraso não vira uma fila de vídeo crescente."""

from queue import Empty, Full


def put_latest(channel, message) -> bool:
    """Substitui a mensagem pendente; nunca espera por um consumidor lento."""
    try:
        channel.put_nowait(message)
        return True
    except Full:
        pass
    try:
        channel.get_nowait()
    except Empty:
        # O feeder pode ainda estar enviando o item. Nesse caso descartamos
        # a entrada nova, em vez de bloquear a captura ou a interface.
        return False
    try:
        channel.put_nowait(message)
        return True
    except Full:
        return False


def receive_latest(channel):
    """Todos os canais têm capacidade 1; uma leitura basta e não espera dados."""
    try:
        return channel.get_nowait()
    except Empty:
        return None
