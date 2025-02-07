SCRIPTS := $(CURDIR)/src/*.py
LOCAL_BIN := $(HOME)/.local/bin

all: migrate

.PHONY: migrate
migrate:
	cp -u $(SCRIPTS) $(LOCAL_BIN)
	rm -f ~/.local/bin/__init__.py
	

