`timescale 1ns/1ps
// Reference-driven testbench for q4_gemv_tile. Vectors come from sw/gen_q4_vectors.py
// (which uses sw/q4_gemv.py's pack_q4), so the nibble packing and Q16.16 scale
// arithmetic are checked against the reference, not hand-written constants.
//   +mem=<words.hex> +act=<acts.hex> +exp=<expected.hex> +rows=N +groups=G
module q4_gemv_tb;
    parameter integer ROWS = 4;
    parameter integer GROUPS = 3;
    localparam integer MEM_WORDS = ROWS * GROUPS * 5;
    localparam integer ACT_BYTES = GROUPS * 32;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic [31:0] row = 32'd0;
    logic [GROUPS*32*8-1:0] activations = '0;
    wire done;
    wire signed [63:0] result;
    wire mem_req;
    wire [31:0] mem_addr;
    wire mem_ready;
    wire [31:0] mem_rdata;

    logic [31:0] mem [0:MEM_WORDS-1];
    logic [7:0] act_bytes [0:ACT_BYTES-1];
    logic [63:0] expected [0:ROWS-1];

    integer r, i, mismatches;
    string fmem, fact, fexp;

    always #5 clk = ~clk;

    assign mem_ready = mem_req;
    assign mem_rdata = (mem_req && (mem_addr[31:2] < MEM_WORDS)) ? mem[mem_addr[31:2]] : 32'd0;

    q4_gemv_tile #(.GROUPS_PER_ROW(GROUPS)) dut (
        .clk, .rst_n, .start, .row, .activations, .done, .result,
        .mem_req, .mem_addr, .mem_ready, .mem_rdata
    );

    initial begin
        if (!$value$plusargs("mem=%s", fmem)) $fatal(1, "need +mem=");
        if (!$value$plusargs("act=%s", fact)) $fatal(1, "need +act=");
        if (!$value$plusargs("exp=%s", fexp)) $fatal(1, "need +exp=");
        $readmemh(fmem, mem);
        $readmemh(fact, act_bytes);
        $readmemh(fexp, expected);

        for (i = 0; i < ACT_BYTES; i = i + 1)
            activations[i*8 +: 8] = act_bytes[i];

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        mismatches = 0;
        for (r = 0; r < ROWS; r = r + 1) begin
            @(negedge clk);
            row = r;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            while (done !== 1'b1) @(negedge clk);
            #1;
            if (result !== expected[r]) begin
                $display("FAIL row=%0d got=%0d expected=%0d", r, result, expected[r]);
                mismatches = mismatches + 1;
            end else begin
                $display("PASS row=%0d result=%0d", r, result);
            end
        end
        if (mismatches == 0) $display("PASS q4_gemv_tb");
        else $display("FAIL q4_gemv_tb (%0d rows)", mismatches);
        $finish;
    end
endmodule
