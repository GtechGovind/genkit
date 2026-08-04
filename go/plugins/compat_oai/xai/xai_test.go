// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package xai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/xai"
)

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("XAI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "sk-should-not-be-used")

	defer func() {
		if got := recover(); got != "xai plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()
	(&xai.XAI{}).Init(context.Background())
}

func TestPluginRegistersModelsAndTranslatesConfig(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/v1/chat/completions")
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}

		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if got := body["model"]; got != xai.ModelGrok3Mini {
			t.Errorf("model = %v, want %q", got, xai.ModelGrok3Mini)
		}
		if got := body["deferred"]; got != true {
			t.Errorf("deferred = %v, want true", got)
		}
		if got := body["reasoning_effort"]; got != "high" {
			t.Errorf("reasoning_effort = %v, want high", got)
		}
		webSearch, ok := body["web_search_options"].(map[string]any)
		if !ok {
			t.Fatalf("web_search_options = %#v, want object", body["web_search_options"])
		}
		if got := webSearch["search_context_size"]; got != "high" {
			t.Errorf("web_search_options.search_context_size = %v, want high", got)
		}
		for _, name := range []string{"reasoningEffort", "webSearchOptions"} {
			if _, ok := body[name]; ok {
				t.Errorf("request contains unconverted %s field", name)
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"chatcmpl-1",
			"object":"chat.completion",
			"created":1,
			"model":"grok-3-mini",
			"choices":[{
				"index":0,
				"message":{"role":"assistant","content":"Grok works"},
				"finish_reason":"stop"
			}],
			"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &xai.XAI{APIKey: "test-key", BaseURL: server.URL + "/v1"}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("xai/"+xai.ModelGrok3Mini),
	)

	if got := plugin.Name(); got != "xai" {
		t.Fatalf("Name() = %q, want xai", got)
	}
	textModels := []string{
		xai.ModelGrok3,
		xai.ModelGrok3Fast,
		xai.ModelGrok3Mini,
		xai.ModelGrok3MiniFast,
	}
	visionModels := []string{xai.ModelGrok2Vision1212}
	for _, group := range []struct {
		models         []string
		wantMedia      bool
		wantMultiturn  bool
		wantSystemRole bool
	}{
		{models: textModels, wantMedia: false, wantMultiturn: true, wantSystemRole: true},
		{models: visionModels, wantMedia: true, wantMultiturn: false, wantSystemRole: false},
	} {
		for _, modelID := range group.models {
			model := plugin.Model(g, modelID)
			if model == nil {
				t.Errorf("Model(%q) = nil", modelID)
				continue
			}
			desc := model.(api.Action).Desc()
			if got, want := desc.Name, "xai/"+modelID; got != want {
				t.Errorf("%s Desc().Name = %q, want %q", modelID, got, want)
			}
			metadata := desc.Metadata["model"].(map[string]any)
			supports := metadata["supports"].(map[string]any)
			for name, check := range map[string]struct {
				got  any
				want any
			}{
				"media":      {got: supports["media"], want: group.wantMedia},
				"multiturn":  {got: supports["multiturn"], want: group.wantMultiturn},
				"systemRole": {got: supports["systemRole"], want: group.wantSystemRole},
				"tools":      {got: supports["tools"], want: true},
				"toolChoice": {got: supports["toolChoice"], want: false},
			} {
				if check.got != check.want {
					t.Errorf("%s %s support = %v, want %v", modelID, name, check.got, check.want)
				}
			}
			output, _ := supports["output"].([]string)
			if !slices.Equal(output, []string{"text", "json"}) {
				t.Errorf("%s output = %v, want [text json]", modelID, output)
			}

			configSchema := metadata["customOptions"].(map[string]any)
			properties := configSchema["properties"].(map[string]any)
			reasoningEffort := properties["reasoningEffort"].(map[string]any)
			enumValues, _ := reasoningEffort["enum"].([]any)
			if !slices.Equal(enumValues, []any{"low", "medium", "high"}) {
				t.Errorf("%s reasoningEffort enum = %v, want [low medium high]", modelID, enumValues)
			}
			if _, ok := properties["deferred"]; !ok {
				t.Errorf("%s config schema is missing deferred", modelID)
			}
			if _, ok := properties["webSearchOptions"]; !ok {
				t.Errorf("%s config schema is missing webSearchOptions", modelID)
			}
		}
	}

	resp, err := genkit.Generate(
		ctx,
		g,
		ai.WithPrompt("Say hi."),
		ai.WithConfig(map[string]any{
			"deferred":        true,
			"reasoningEffort": "high",
			"webSearchOptions": map[string]any{
				"search_context_size": "high",
			},
		}),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := resp.Text(); got != "Grok works" {
		t.Fatalf("Text() = %q, want %q", got, "Grok works")
	}
	if requests != 1 {
		t.Fatalf("requests = %d, want 1", requests)
	}
}
