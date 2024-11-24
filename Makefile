SCRIPTS := $(CURDIR)/src/*.py
LOCAL_BIN := $(HOME)/.local/bin

all: migrate

.PHONY: migrate
migrate:
	mv -u $(SCRIPTS) $(LOCAL_BIN)
	

