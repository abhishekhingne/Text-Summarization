from build_data import BuildData
from seq2seq import TextSummarization
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_hidden", type=int, default=150, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Network depth.")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam width for beam search decoder.")
    parser.add_argument("--embedding_size", type=int, default=300, help="Word embedding size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--keep_prob", type=float, default=0.8, help="Dropout keep prob.")
    parser.add_argument("--mode", type=str, default="train", help="Train or Test")
    parser.add_argument("--with_model", action="store_true", help="Continue from previously saved model")

    args = parser.parse_args()
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    beam_width = args.beam_width
    embedding_size = args.embedding_size
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    keep_prob = args.keep_prob
    mode = args.mode
    with_model = args.with_model
    data = BuildData()
    text_summarization = TextSummarization(num_hidden=num_hidden, num_layers=num_layers, beam_width=beam_width,
                                           embedding_size=embedding_size, learning_rate=learning_rate,
                                           batch_size=batch_size, num_epochs=num_epochs, keep_prob=keep_prob,
                                           with_model=False)
    if mode == "train":
        data.load_data()
        data.build_dict()
        train_x, train_y = data.build_dataset
        text_summarization.start_training(train_x=train_x, train_y=train_y, word_dict=data.word_dict,
                                          reversed_word_dict=data.reversed_word_dict)
