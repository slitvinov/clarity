BIN = $(HOME)/bin # where to install
PROGS = ugrep     # what to install
install:; mkdir -p $(BIN) && cp $(PROGS) $(BIN)
.PHONY: install
